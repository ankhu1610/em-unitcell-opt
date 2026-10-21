"""Independent pure-PWE diagnostic; NumPy only, no fitting or COMSOL calls.

The boundary integral represents the documented disk/triangle intersection.
Numerical quadrature of circular arcs is refined separately from the PWE basis.
Existing production code and experimental results are not overwritten.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT.parent / 'ML files' / 'COMSOL_ML_Handoff_20261008'
C = 299792458.0


def reciprocal_indices(cutoff):
    m, n = np.meshgrid(np.arange(-cutoff, cutoff + 1),
                       np.arange(-cutoff, cutoff + 1), indexing='ij')
    return np.column_stack((m.ravel(), n.ravel()))


def reciprocal_basis(a):
    return 2 * np.pi / a * np.array([[1., -1 / np.sqrt(3)],
                                    [0., 2 / np.sqrt(3)]])


def path_coordinates(k, a):
    k = np.asarray(k)
    alpha = np.where(k < 1, k / 2, np.where(k < 2, (k + 2) / 6, 2 * (3-k) / 3))
    beta = np.where(k < 1, k / 2, np.where(k < 2, (4-k) / 6, (3-k) / 3))
    return np.column_stack((alpha, beta)) @ reciprocal_basis(a)


def hole_ft(q, side, up=True, order=64):
    """Unnormalised integral exp(-i q.r) dA for a centred clipped triangle.

    Divergence theorem: integral = i/q^2 * boundary integral (q.n) exp(-iq.r) ds.
    Straight chords are integrated analytically; circular arcs by Gauss-Legendre.
    Lengths may use any common unit with q in its inverse.
    """
    q = np.asarray(q, dtype=float) * (1 if up else -1)
    r = side / (1.5 * np.sqrt(3))
    h = side / (2 * np.sqrt(3))
    s = np.sqrt(r*r - h*h)
    gamma = np.arccos(h/r)
    area = side**2 * (4*np.pi/27 - 4*np.arccos(.75)/9 + np.sqrt(7)/12)
    norm2 = np.sum(q*q, axis=1)
    result = np.full(len(q), area, dtype=complex)
    mask = norm2 > 0
    qq = q[mask]
    boundary = np.zeros(len(qq), dtype=complex)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for angle in [-np.pi/2, np.pi/6, 5*np.pi/6]:
        normal = np.array([np.cos(angle), np.sin(angle)])
        tangent = np.array([-normal[1], normal[0]])
        qn, qt = qq @ normal, qq @ tangent
        boundary += qn * (2*s) * np.sinc(qt*s/np.pi) * np.exp(-1j*qn*h)
        lo, hi = angle+gamma, angle+2*np.pi/3-gamma
        theta = (hi+lo)/2 + nodes*(hi-lo)/2
        normals = np.column_stack((np.cos(theta), np.sin(theta)))
        proj = qq @ normals.T
        boundary += r * (hi-lo)/2 * np.sum(
            proj * np.exp(-1j*r*proj) * weights, axis=1)
    result[mask] = 1j * boundary / norm2[mask]
    return result


def eta_matrix(indices, a, sides, order=64):
    # Unique difference coefficients avoid repeated boundary integration.
    differences = indices[:, None, :] - indices[None, :, :]
    unique, inverse = np.unique(differences.reshape(-1, 2), axis=0, return_inverse=True)
    q = unique @ reciprocal_basis(a)
    centers = np.array([[a, a/np.sqrt(3)], [a/2, a/(2*np.sqrt(3))]])
    coeff = np.zeros(len(unique), dtype=complex)
    area = np.sqrt(3)/2*a*a
    for side, center, up in zip(sides, centers, [False, True]):
        coeff += hole_ft(q, side, up, order) * np.exp(-1j*(q @ center)) / area
    coeff *= 1 - 1/9
    coeff[np.all(unique == 0, axis=1)] += 1/9
    return coeff[inverse].reshape(len(indices), len(indices))


def solve(indices, a, eta, kpoints, n_bands=6):
    g = indices @ reciprocal_basis(a)
    frequencies = []
    for k in kpoints:
        kg = g+k
        operator = eta * (kg @ kg.T)
        eigenvalues = np.linalg.eigvalsh(operator)
        scale = np.max(np.abs(eigenvalues))
        if eigenvalues[0] < -1e-11*scale:
            raise ValueError('Materially negative eigenvalue')
        frequencies.append(C/(2*np.pi*1e9) * np.sqrt(np.maximum(eigenvalues[:n_bands], 0)))
    return np.array(frequencies)


def source_designs():
    with (HANDOFF/'design_inventory.csv').open(encoding='utf-8-sig', newline='') as f:
        inventory = list(csv.DictReader(f))
    with (HANDOFF/'saved_eigenfrequencies.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    lookup = {(r['design_id'], round(float(r['k']), 2), int(r['stored_mode_index'])):
              float(r['frequency_real_GHz']) for r in records}
    return inventory, lookup


def run(args):
    t0 = time.perf_counter()
    a = 242.5  # micrometres; eigenvalues converted to SI explicitly in solve calls
    inventory, lookup = source_designs()
    data = np.load(ROOT/'data/processed/bands.npz')
    metadata = json.loads((HANDOFF/'geometry_metadata.json').read_text())
    checks = {}
    # Verify homogeneous analytical spectrum for an arbitrary off-symmetry k.
    ids = reciprocal_indices(3)
    kk = path_coordinates([.37], a*1e-6)
    eta = np.eye(len(ids))/9
    predicted = solve(ids, a*1e-6, eta, kk)
    expected = np.sort(C*np.linalg.norm(ids@reciprocal_basis(a*1e-6)+kk[0], axis=1)
                       /(2*np.pi*3*1e9))[:6]
    checks['homogeneous_max_abs_GHz'] = float(np.max(np.abs(predicted[0]-expected)))
    q = np.array([[0,0], [1,0], [0,1], [1,2], [-2,1]]) @ reciprocal_basis(a)
    ft = hole_ft(q, 140.00744027848427)
    checks['boundary_ft_conjugacy_error'] = float(np.max(np.abs(hole_ft(-q, 140.00744027848427)-ft.conj())))
    checks['zero_fourier_area_error_um2'] = float(abs(ft[0]-metadata['area_coefficient']*140.00744027848427**2))
    # Independent dense raster check of low Fourier coefficients, not an eigensolve.
    N = 1024
    u,v = np.meshgrid(np.arange(N)/N, np.arange(N)/N, indexing='ij')
    x,y = a*(u+v/2), a*np.sqrt(3)/2*v
    inv_eps = np.full_like(x, 1/9)
    sides = [140.00744027848427, 91.00483618101477]
    for side, center, up in zip(sides, [(a,a/np.sqrt(3)), (a/2,a/(2*np.sqrt(3)))], [False,True]):
        mask = np.zeros_like(x, dtype=bool)
        for du in [-1,0,1]:
            for dv in [-1,0,1]:
                dx = x-(center[0]+du*a+dv*a/2)
                dy = y-(center[1]+dv*a*np.sqrt(3)/2)
                if not up:
                    dx,dy = -dx,-dy
                height = np.sqrt(3)/2*side
                tri = (dy>=-height/3)&(dy-np.sqrt(3)*dx<=2*height/3)&(dy+np.sqrt(3)*dx<=2*height/3)
                mask |= tri & (dx*dx+dy*dy <= (side/(1.5*np.sqrt(3)))**2)
        inv_eps[mask] = 1
    fft = np.fft.fft2(inv_eps)/N**2
    small = reciprocal_indices(2)
    boundary = eta_matrix(small, a, sides)
    diff = small[:,None,:]-small[None,:,:]
    raster = fft[diff[:,:,0]%N,diff[:,:,1]%N]
    checks['boundary_vs_1024_raster_max_abs_eta'] = float(np.max(np.abs(boundary-raster)))
    checks['hermitian_max_abs_eta'] = float(np.max(np.abs(boundary-boundary.conj().T)))
    negative = solve(small, a*1e-6, boundary, -kk)
    positive = solve(small, a*1e-6, boundary, kk)
    checks['time_reversal_max_abs_GHz'] = float(np.max(np.abs(negative-positive)))
    # Source vs processed tensor alignment across all designs, with exact source lengths.
    alignment = []
    for struct in range(2):
        errs=[]
        for d in inventory:
            i=int(np.argmin(abs(data['L1_um']-float(d['L1_um']))))
            j=int(np.argmin(abs(data['L2_um']-float(d['L2_um']))))
            source=np.array([[lookup[d['design_id'],round(float(k),2),b] for b in [1,2]] for k in data['k']])
            errs.append(np.abs(np.sort(source,axis=1)-data['F'][struct,i,j]))
        alignment.append({'structure':['para','r1'][struct], 'mae_GHz':float(np.mean(errs)),
                          'max_abs_GHz':float(np.max(errs))})
    with (HANDOFF/'bloch_path.csv').open(encoding='utf-8-sig') as f:
        path=list(csv.DictReader(f))
    actual=path_coordinates([float(r['k']) for r in path],a*1e-6)
    saved=np.array([[float(r['kx_rad_per_m']),float(r['ky_rad_per_m'])] for r in path])
    checks['source_kpath_max_abs_rad_per_m']=float(np.max(np.abs(actual-saved)))
    result={'method':'pure TE Hz PWE; boundary quadrature; no fitted constants or residual',
            'source_metadata_sha256':hashlib.sha256((HANDOFF/'geometry_metadata.json').read_bytes()).hexdigest(),
            'checks':checks,'source_processed_alignment':alignment,'designs':[]}
    for design_id in args.designs:
        d=next(row for row in inventory if row['design_id']==design_id)
        sides=[float(d['L1_um']),float(d['L2_um'])]
        target=np.array([[lookup[design_id,k,b] for b in [1,2]] for k in [1.,2.]])
        item={'design_id':design_id,'sides_um':sides,'source_M_K_GHz':target.tolist(),'runs':[]}
        print(design_id, 'source M,K',target.tolist(),flush=True)
        last=None
        for cutoff in args.cutoffs:
            start=time.perf_counter()
            indices=reciprocal_indices(cutoff)
            eta=eta_matrix(indices,a,sides,args.order)
            f=solve(indices,a*1e-6,eta,path_coordinates([1.,2.],a*1e-6))
            err=np.abs(f[:,:2]-target)
            run={'M':cutoff,'plane_waves':len(indices),'arc_quadrature_order':args.order,
                 'PWE_M_K_first6_GHz':f.tolist(),'mae_GHz':float(err.mean()),
                 'max_relative_frequency_error_pct':float(np.max(err/target)*100),
                 'direct_K_GHz':float(f[1,1]-f[1,0]),
                 'indirect_Mupper_Klower_GHz':float(f[0,1]-f[1,0]),
                 'source_direct_K_GHz':float(target[1,1]-target[1,0]),
                 'source_indirect_Mupper_Klower_GHz':float(target[0,1]-target[1,0]),
                 'seconds':time.perf_counter()-start}
            if last is not None:
                run['max_relative_change_from_previous_pct']=float(np.max(abs(f[:,:2]-last)/f[:,:2])*100)
            item['runs'].append(run)
            last=f[:,:2]
            print('  ',cutoff, 'bands',f[:,:2].tolist(),'MAE',run['mae_GHz'],
                  'max %',run['max_relative_frequency_error_pct'],'seconds',run['seconds'],flush=True)
        # Independent convergence of boundary quadrature at fixed largest basis.
        eta_ref=eta_matrix(indices,a,sides,args.order*2)
        result_eta=float(np.max(np.abs(eta_ref-eta)))
        item['quadrature_doubling_max_abs_eta']=result_eta
        # Opposite Fourier sign must give the same spectrum for real media.
        inv_f=solve(indices,a*1e-6,eta.conj(),path_coordinates([1.,2.],a*1e-6))
        item['opposite_FT_sign_max_abs_GHz']=float(np.max(abs(inv_f-f)))
        result['designs'].append(item)
    result['elapsed_seconds']=time.perf_counter()-t0
    write_report(result)
    print('CHECKS',json.dumps(checks),flush=True)
    print('ALIGNMENT',json.dumps(alignment),flush=True)


def write_report(result):
    """All reported measurements are generated from the numerical result object."""
    out=ROOT/'results'/'pure_pwe_audit'
    out.mkdir(parents=True,exist_ok=True)
    for d in result['designs']:
        target=np.array(d['source_M_K_GHz'])
        for r in d['runs']:
            r['source_direct_K_GHz']=float(target[1,1]-target[1,0])
            r['source_indirect_Mupper_Klower_GHz']=float(target[0,1]-target[1,0])
            r['direct_K_abs_error_GHz']=abs(r['direct_K_GHz']-r['source_direct_K_GHz'])
            r['indirect_Mupper_Klower_abs_error_GHz']=abs(r['indirect_Mupper_Klower_GHz']-r['source_indirect_Mupper_Klower_GHz'])
        first=np.array(d['runs'][0]['PWE_M_K_first6_GHz'])
        d['all_tested_basis_increments_nonincreasing']=all(
            np.all(np.array(new['PWE_M_K_first6_GHz']) <= np.array(old['PWE_M_K_first6_GHz'])+1e-8)
            for old,new in zip(d['runs'][:-1],d['runs'][1:]))
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Pure PWE diagnostic','', 'Generated by scripts/audit_pure_pwe.py; no fitted correction.','',
           'Boundary quadrature and a NumPy eigensolver provide an independent implementation.',
           'This tests numerical convergence and agreement with saved spectra, not fresh FEM convergence.','',
           '## Checks','', '```json',json.dumps(result['checks'],indent=2),'```','',
           'The zero-coefficient area check is algebraic, using the same analytic area expression.',
           'The dense raster comparison independently checks the nonzero Fourier coefficients.','',
           '## Source / processed alignment','', '```json',
           json.dumps(result['source_processed_alignment'],indent=2),'```','',
           'The recovered source closely matches para; it cannot establish the physical provenance of r1.',
           'Exact saved geometry controls are used rather than rounded processed grid labels.','',
           '## M/K frequency comparison','',
           '|Design|M|Plane waves|MAE GHz|Maximum frequency error %|Change from preceding M %|',
           '|---|---:|---:|---:|---:|---:|']
    for d in result['designs']:
        for r in d['runs']:
            change=r.get('max_relative_change_from_previous_pct',float('nan'))
            lines.append(f"|{d['design_id']}|{r['M']}|{r['plane_waves']}|{r['mae_GHz']:.6f}|{r['max_relative_frequency_error_pct']:.6f}|{change:.6f}|")
    lines += ['', '## Largest tested basis: local gap errors','',
              '|Design|PWE direct K GHz|Source direct K GHz|Absolute error GHz|PWE indirect M/K GHz|Source indirect M/K GHz|Absolute error GHz|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for d in result['designs']:
        r=d['runs'][-1]
        lines.append(f"|{d['design_id']}|{r['direct_K_GHz']:.6f}|{r['source_direct_K_GHz']:.6f}|{r['direct_K_abs_error_GHz']:.6f}|{r['indirect_Mupper_Klower_GHz']:.6f}|{r['source_indirect_Mupper_Klower_GHz']:.6f}|{r['indirect_Mupper_Klower_abs_error_GHz']:.6f}|")
    lines += ['', 'These are selected-point separations, not a full-zone or full-path gap claim.','',
              '## Interpretation','',
              'The tested pure-PWE frequencies continue approaching the recovered source as the nested basis grows.',
              'The old scripts do not establish a universal PWE accuracy plateau: they use obsolete geometry/material assumptions, stale imports, and inconsistent parameter ordering.',
              'The old Gate P report also asserts a geometry explanation without an isolating experiment.',
              'Its final path segment ends at a reciprocal-equivalent Gamma; this is not by itself proof of the cause of a discrepancy.',
              'Both the continuum problem and numerical convergence must match before asserting FEM/PWE equivalence.',
              'Quadrature-doubling errors and first six eigenfrequencies are retained in results.json.',
              'A small final basis increment is evidence about convergence, not a certified error bound.',
              'Only the listed geometries and M/K points were compared. Fresh FEM mesh convergence, the full path, and all geometries were not established.',
              'NumPy was available in the bundled runtime; torch/scipy/pytest were not. This independent diagnostic does not execute or certify the production PyTorch implementation.',
              '', '## Recommended next pure-PWE work','',
              '1. Consolidate the physical geometry, exact parameter values, and k-path into a single source.',
              '2. Keep boundary quadrature and basis convergence separate; retain an independent reference implementation.',
              '3. Investigate documented interface-aware Fourier factorization or an independent established plane-wave solver to accelerate high-contrast convergence.',
              '4. Validate additional k-points and geometries with the same settings before choosing a production accuracy/runtime tradeoff.',
              '5. Establish reference FEM uncertainty before interpreting agreement as physical accuracy.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cutoffs',type=int,nargs='+',default=[3,5,7,9,12,15])
    parser.add_argument('--designs',nargs='+',default=['D001','D028','D064'])
    parser.add_argument('--order',type=int,default=64)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        write_report(json.loads((ROOT/'results/pure_pwe_audit/results.json').read_text()))
    else:
        run(args)
