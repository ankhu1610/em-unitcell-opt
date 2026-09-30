"""
details.py
==========
Comprehensive Dataset Intelligence Report Generator
for COMSOL Electromagnetic Simulation Data

Files analyzed:
    - para.csv  (~3.9 GB, ~29.5M data rows)  — parametric_sweep_UC.mph
    - r1.csv    (~4.2 GB, ~31.9M data rows)  — (no model header)

Both files contain 2D FEM mesh node-level electromagnetic
simulation results from COMSOL 6.4, with parametric sweeps
over geometric parameters (L1, L2) and a wavenumber (k).

Output:
    reports/dataset_details_report.md   — LLM-ready Markdown
    reports/dataset_details_report.json — structured JSON

Run:
    python details.py
"""

import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FILES = {
    "para": BASE_DIR / "para.csv",
    "r1":   BASE_DIR / "r1.csv",
}
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_MD  = OUTPUT_DIR / "dataset_details_report.md"
OUTPUT_JSON = OUTPUT_DIR / "dataset_details_report.json"

# How many data lines to read for statistical sampling
# (reading all 30M+ lines for stats is feasible but slow;
#  we do a FULL PASS for parameter discovery + line count,
#  and a SAMPLED pass for statistics.)
STAT_SAMPLE_SIZE = 2_000_000   # 2M lines per file for stats
FULL_PASS = True               # full pass for parameter discovery


# ============================================================
# Parsing helpers
# ============================================================

def parse_header(filepath):
    """Read COMSOL %-prefixed header lines."""
    meta = {}
    col_line = None
    header_count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("%"):
                break
            header_count += 1
            stripped = line.lstrip("% ").strip()
            if "," in stripped:
                parts = stripped.split(",", 1)
                key = parts[0].strip()
                val = parts[1].strip().strip('"')
                meta[key] = val
    # The last header line starting with "% X" is the column spec
    if "X" in meta:
        # The column header is: % X,Y,L1,L2,k,lambda,emw.freq (GHz)
        col_line = meta.get("X", "")
        # Reconstruct
        cols = ["X"]
        if col_line:
            cols += [c.strip() for c in col_line.split(",")]
        meta["columns"] = cols
    else:
        # fallback: re-read
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("% X"):
                    raw = line.lstrip("% ").strip()
                    meta["columns"] = [c.strip() for c in raw.split(",")]
                    break
    meta["header_line_count"] = header_count
    return meta


def parse_complex(s):
    """Parse a COMSOL complex number string like '4.96E11+75844.6i'."""
    s = s.strip()
    if "i" in s:
        # Complex: format is  real_part+imag_parti  or real_part-imag_parti
        s = s.rstrip("i")
        # find the + or - that separates real and imaginary
        # but NOT the E+/E- in scientific notation
        idx = -1
        for i in range(len(s) - 1, 0, -1):
            if s[i] in ("+", "-") and s[i-1] not in ("E", "e"):
                idx = i
                break
        if idx > 0:
            real_part = float(s[:idx])
            imag_part = float(s[idx:])
            return complex(real_part, imag_part)
        else:
            return complex(0, float(s))
    else:
        return complex(float(s), 0)


def safe_float(s):
    """Parse a string to float, handling complex COMSOL values."""
    s = s.strip()
    if "i" in s:
        c = parse_complex(s)
        return abs(c)  # magnitude for statistics
    return float(s)


# ============================================================
# Full-pass: line count + parameter discovery
# ============================================================

def full_pass_analysis(filepath, label):
    """Single pass through entire file to count lines and discover parameters."""
    print(f"  [{label}] Full pass: counting lines and discovering parameters...")

    meta = parse_header(filepath)
    header_lines = meta.get("header_line_count", 0)

    total_lines = 0
    data_lines = 0

    # Parameter collectors
    l1_vals = set()
    l2_vals = set()
    k_vals = set()
    freq_vals = set()
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    freq_min, freq_max = float("inf"), float("-inf")

    complex_lambda_count = 0
    real_lambda_count = 0

    xy_pairs = set()
    xy_count_per_params = Counter()  # (L1, L2, k) -> count of unique XY

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            if line.startswith("%"):
                continue
            data_lines += 1

            parts = line.strip().split(",")
            if len(parts) < 7:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                l1 = parts[2].strip()
                l2 = parts[3].strip()
                k = parts[4].strip()
                lam_str = parts[5].strip()
                freq_str = parts[6].strip()

                l1_vals.add(l1)
                l2_vals.add(l2)
                k_vals.add(k)

                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

                try:
                    freq = float(freq_str)
                    freq_vals.add(round(freq, 6))
                    freq_min = min(freq_min, freq)
                    freq_max = max(freq_max, freq)
                except ValueError:
                    pass

                if "i" in lam_str:
                    complex_lambda_count += 1
                else:
                    real_lambda_count += 1

                # Track unique XY per first 5M lines only (memory)
                if data_lines <= 5_000_000:
                    xy_pairs.add((parts[0].strip(), parts[1].strip()))

            except (ValueError, IndexError):
                continue

            if data_lines % 5_000_000 == 0:
                print(f"    ... {data_lines:,} data lines processed")

    # Sort k values numerically
    k_float = sorted([float(k) for k in k_vals])

    result = {
        "file": str(filepath.name),
        "file_size_gb": round(os.path.getsize(filepath) / (1024**3), 3),
        "total_lines": total_lines,
        "header_lines": header_lines,
        "data_lines": data_lines,
        "comsol_metadata": {
            k: v for k, v in meta.items()
            if k not in ("columns", "header_line_count")
        },
        "columns": meta.get("columns", []),
        "geometry": {
            "dimension": meta.get("Dimension", "2"),
            "length_unit": meta.get("Length unit", "unknown"),
            "x_range": [round(x_min, 6), round(x_max, 6)],
            "y_range": [round(y_min, 6), round(y_max, 6)],
            "unique_xy_pairs_in_first_5M": len(xy_pairs),
        },
        "parametric_sweep": {
            "L1": {
                "count": len(l1_vals),
                "values_um": sorted(l1_vals),
                "values_nm": [round(float(v) * 1e6, 2) for v in sorted(l1_vals)],
                "interpretation": "Geometric parameter (likely unit cell length L1)",
            },
            "L2": {
                "count": len(l2_vals),
                "values_um": sorted(l2_vals),
                "values_nm": [round(float(v) * 1e6, 2) for v in sorted(l2_vals)],
                "interpretation": "Geometric parameter (likely unit cell length L2)",
            },
            "k": {
                "count": len(k_float),
                "min": k_float[0] if k_float else None,
                "max": k_float[-1] if k_float else None,
                "step": round(k_float[1] - k_float[0], 6) if len(k_float) > 1 else None,
                "interpretation": "Normalized wave-vector (Bloch k) for band-structure sweep",
            },
        },
        "frequency_output": {
            "column": "emw.freq (GHz)",
            "unit": "GHz",
            "unique_count": len(freq_vals),
            "min_ghz": round(freq_min, 4) if freq_min != float("inf") else None,
            "max_ghz": round(freq_max, 4) if freq_max != float("inf") else None,
            "interpretation": "Eigenfrequency solutions at each (L1,L2,k) point",
        },
        "lambda_column": {
            "real_count": real_lambda_count,
            "complex_count": complex_lambda_count,
            "total": real_lambda_count + complex_lambda_count,
            "complex_fraction": round(
                complex_lambda_count / max(1, real_lambda_count + complex_lambda_count), 4
            ),
            "interpretation": (
                "Eigenvalue λ. Real = propagating mode, "
                "Complex = evanescent/lossy mode. "
                "Imaginary part represents attenuation/loss."
            ),
        },
    }

    print(f"  [{label}] Done: {data_lines:,} data lines, "
          f"{len(k_float)} k-values, {len(freq_vals)} unique frequencies")

    return result


# ============================================================
# Statistical sampling pass
# ============================================================

def statistical_analysis(filepath, label, sample_size=2_000_000):
    """Read a sample for statistical distributions."""
    print(f"  [{label}] Statistical sampling ({sample_size:,} lines)...")

    # Collect arrays
    x_arr = []
    y_arr = []
    k_arr = []
    freq_arr = []
    lambda_mag_arr = []
    lambda_real_arr = []
    lambda_imag_arr = []

    count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                k = float(parts[4])
                freq = float(parts[6])
                lam_str = parts[5].strip()

                x_arr.append(x)
                y_arr.append(y)
                k_arr.append(k)
                freq_arr.append(freq)

                if "i" in lam_str:
                    c = parse_complex(lam_str)
                    lambda_mag_arr.append(abs(c))
                    lambda_real_arr.append(c.real)
                    lambda_imag_arr.append(c.imag)
                else:
                    val = float(lam_str)
                    lambda_mag_arr.append(abs(val))
                    lambda_real_arr.append(val)
                    lambda_imag_arr.append(0.0)

                count += 1
            except (ValueError, IndexError):
                continue

            if count >= sample_size:
                break

    # Convert to numpy
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    k_arr = np.array(k_arr)
    freq_arr = np.array(freq_arr)
    lambda_mag = np.array(lambda_mag_arr)
    lambda_real = np.array(lambda_real_arr)
    lambda_imag = np.array(lambda_imag_arr)

    def col_stats(arr, name):
        if len(arr) == 0:
            return {"name": name, "count": 0}
        return {
            "name": name,
            "count": int(len(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": round(float(np.nanmean(arr)), 6),
            "std": round(float(np.nanstd(arr)), 6),
            "median": round(float(np.nanmedian(arr)), 6),
            "q25": round(float(np.nanpercentile(arr, 25)), 6),
            "q75": round(float(np.nanpercentile(arr, 75)), 6),
            "skewness": round(float(skew_calc(arr)), 6),
            "nan_count": int(np.isnan(arr).sum()),
            "zero_count": int((arr == 0).sum()),
            "negative_count": int((arr < 0).sum()),
        }

    def skew_calc(arr):
        n = len(arr)
        if n < 3:
            return 0.0
        m = np.mean(arr)
        s = np.std(arr)
        if s == 0:
            return 0.0
        return float(np.mean(((arr - m) / s) ** 3))

    # Frequency distribution: bin into GHz bands
    freq_hist_bins = np.arange(0, 600, 10)  # 10 GHz bins
    freq_hist, _ = np.histogram(freq_arr, bins=freq_hist_bins)
    freq_distribution = {
        f"{int(freq_hist_bins[i])}-{int(freq_hist_bins[i+1])} GHz": int(freq_hist[i])
        for i in range(len(freq_hist)) if freq_hist[i] > 0
    }

    # K distribution
    k_hist_bins = np.arange(0, 3.1, 0.1)
    k_hist, _ = np.histogram(k_arr, bins=k_hist_bins)
    k_distribution = {
        f"{k_hist_bins[i]:.1f}-{k_hist_bins[i+1]:.1f}": int(k_hist[i])
        for i in range(len(k_hist)) if k_hist[i] > 0
    }

    # Correlation: freq vs k
    if len(freq_arr) > 10 and len(k_arr) > 10:
        corr_freq_k = round(float(np.corrcoef(freq_arr, k_arr)[0, 1]), 6)
    else:
        corr_freq_k = None

    # Correlation: freq vs lambda_magnitude
    if len(freq_arr) > 10 and len(lambda_mag) > 10:
        # Filter out extreme values for correlation
        mask = np.isfinite(lambda_mag) & np.isfinite(freq_arr)
        if mask.sum() > 10:
            corr_freq_lam = round(float(np.corrcoef(
                freq_arr[mask], lambda_mag[mask]
            )[0, 1]), 6)
        else:
            corr_freq_lam = None
    else:
        corr_freq_lam = None

    # Band count per k-value (eigenfrequencies per k)
    k_to_freqs = defaultdict(set)
    for i in range(min(len(k_arr), 500000)):
        k_to_freqs[round(k_arr[i], 4)].add(round(freq_arr[i], 4))
    bands_per_k = [len(v) for v in k_to_freqs.values()]

    # Nodes per (L1,L2,k,freq) — how many spatial mesh nodes per eigenmode
    # This tells us the FEM mesh density

    result = {
        "sample_size": count,
        "column_statistics": {
            "X": col_stats(x_arr, "X (µm)"),
            "Y": col_stats(y_arr, "Y (µm)"),
            "k": col_stats(k_arr, "k (normalized wave-vector)"),
            "frequency": col_stats(freq_arr, "emw.freq (GHz)"),
            "lambda_magnitude": col_stats(lambda_mag, "|λ| (eigenvalue magnitude)"),
            "lambda_real": col_stats(lambda_real, "Re(λ)"),
            "lambda_imag": col_stats(lambda_imag, "Im(λ)"),
        },
        "frequency_distribution_10GHz_bins": freq_distribution,
        "k_distribution": k_distribution,
        "correlations": {
            "freq_vs_k": corr_freq_k,
            "freq_vs_lambda_magnitude": corr_freq_lam,
        },
        "eigenmode_analysis": {
            "avg_bands_per_k": round(np.mean(bands_per_k), 2) if bands_per_k else None,
            "min_bands_per_k": min(bands_per_k) if bands_per_k else None,
            "max_bands_per_k": max(bands_per_k) if bands_per_k else None,
            "interpretation": (
                "Number of eigenfrequency solutions found per k-value. "
                "This represents the number of bands in the photonic band structure."
            ),
        },
    }

    print(f"  [{label}] Stats done: {count:,} lines sampled")
    return result


# ============================================================
# Cross-file comparison
# ============================================================

def cross_file_analysis(para_full, r1_full):
    """Compare the two datasets."""
    print("  Cross-file analysis...")

    # Column comparison
    para_cols = para_full.get("columns", [])
    r1_cols = r1_full.get("columns", [])

    # Parameter comparison
    para_k = para_full["parametric_sweep"]["k"]
    r1_k = r1_full["parametric_sweep"]["k"]

    para_l2 = set(para_full["parametric_sweep"]["L2"]["values_um"])
    r1_l2 = set(r1_full["parametric_sweep"]["L2"]["values_um"])

    result = {
        "schema_comparison": {
            "para_columns": para_cols,
            "r1_columns": r1_cols,
            "identical_schema": para_cols == r1_cols,
            "common_columns": list(set(para_cols) & set(r1_cols)),
        },
        "size_comparison": {
            "para_data_lines": para_full["data_lines"],
            "r1_data_lines": r1_full["data_lines"],
            "ratio_r1_to_para": round(
                r1_full["data_lines"] / max(1, para_full["data_lines"]), 4
            ),
        },
        "parameter_comparison": {
            "L1_same": (
                para_full["parametric_sweep"]["L1"]["values_um"]
                == r1_full["parametric_sweep"]["L1"]["values_um"]
            ),
            "L2_same": para_l2 == r1_l2,
            "L2_common": sorted(para_l2 & r1_l2),
            "L2_only_in_para": sorted(para_l2 - r1_l2),
            "L2_only_in_r1": sorted(r1_l2 - para_l2),
            "k_range_same": (
                para_k["min"] == r1_k["min"]
                and para_k["max"] == r1_k["max"]
                and para_k["count"] == r1_k["count"]
            ),
            "k_count_para": para_k["count"],
            "k_count_r1": r1_k["count"],
        },
        "geometry_comparison": {
            "para_x_range": para_full["geometry"]["x_range"],
            "r1_x_range": r1_full["geometry"]["x_range"],
            "para_y_range": para_full["geometry"]["y_range"],
            "r1_y_range": r1_full["geometry"]["y_range"],
            "para_unique_xy_5M": para_full["geometry"]["unique_xy_pairs_in_first_5M"],
            "r1_unique_xy_5M": r1_full["geometry"]["unique_xy_pairs_in_first_5M"],
            "interpretation": (
                "Different XY pairs and/or ranges suggest different mesh geometries. "
                "para.csv comes from parametric_sweep_UC.mph (Unit Cell model). "
                "r1.csv may represent a different geometry, mesh, or simulation run."
            ),
        },
        "likely_relationship": {
            "hypothesis": (
                "Both files appear to be COMSOL eigenfrequency simulation outputs "
                "for electromagnetic (microwave/THz) unit cell structures. "
                "They share the same parametric sweep parameters (L1, L2, k) "
                "and output columns (X, Y, lambda, freq). "
                "The difference in row count and XY coordinates suggests they use "
                "different FEM meshes or represent different geometries/designs. "
                "This is consistent with an INVERSE DESIGN study where multiple "
                "unit cell geometries are explored."
            ),
            "are_train_test": False,
            "are_different_designs": True,
            "share_parameter_space": True,
        },
    }

    return result


# ============================================================
# Data quality checks
# ============================================================

def data_quality_analysis(filepath, label, sample_size=500_000):
    """Check data quality issues."""
    print(f"  [{label}] Data quality checks...")

    parse_errors = 0
    short_lines = 0
    nan_freq = 0
    negative_freq = 0
    duplicate_adjacent = 0
    prev_line = None
    total = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("%"):
                continue
            total += 1
            stripped = line.strip()
            parts = stripped.split(",")

            if len(parts) < 7:
                short_lines += 1
                continue

            try:
                freq = float(parts[6])
                if freq < 0:
                    negative_freq += 1
                if math.isnan(freq):
                    nan_freq += 1
            except ValueError:
                parse_errors += 1

            if stripped == prev_line:
                duplicate_adjacent += 1
            prev_line = stripped

            if total >= sample_size:
                break

    result = {
        "lines_checked": total,
        "parse_errors": parse_errors,
        "short_lines_under_7_fields": short_lines,
        "nan_frequency": nan_freq,
        "negative_frequency": negative_freq,
        "adjacent_duplicate_lines": duplicate_adjacent,
        "parse_error_rate": round(parse_errors / max(1, total) * 100, 4),
        "quality_assessment": (
            "CLEAN" if (parse_errors + short_lines + nan_freq) == 0
            else "MINOR ISSUES" if (parse_errors + short_lines) < 100
            else "NEEDS CLEANING"
        ),
    }

    print(f"  [{label}] Quality: {result['quality_assessment']}")
    return result


# ============================================================
# Physics interpretation
# ============================================================

def physics_interpretation(para_full, r1_full, para_stats, r1_stats):
    """Generate domain-specific physics interpretation."""
    return {
        "domain": "Computational Electromagnetics / Photonic Crystal Design",
        "simulation_software": "COMSOL Multiphysics 6.4 (EMW module)",
        "problem_type": "Eigenfrequency analysis of 2D electromagnetic unit cells",
        "context": (
            "This dataset comes from parametric sweeps of 2D photonic/metamaterial "
            "unit cells. The simulation solves Maxwell's equations in eigenvalue form "
            "to find resonant frequencies (photonic band structure) as a function of "
            "Bloch wave-vector k and geometric parameters L1, L2."
        ),
        "columns_explained": {
            "X": "Horizontal coordinate of FEM mesh node (µm)",
            "Y": "Vertical coordinate of FEM mesh node (µm)",
            "L1": (
                "First geometric parameter of unit cell (140-189 nm, 8 values). "
                "Likely the lattice constant or a structural dimension."
            ),
            "L2": (
                "Second geometric parameter of unit cell (91-140 nm, 8 values). "
                "Swept across 8 values — likely a feature size (hole radius, slab thickness, etc.)."
            ),
            "k": (
                "Normalized Bloch wave-vector, swept 0.1 to 2.9 in steps of 0.01 (281 values). "
                "This traces the photonic band structure along the Brillouin zone path."
            ),
            "lambda": (
                "Eigenvalue from COMSOL. Real part = resonant frequency parameter. "
                "Imaginary part (when present) = loss/attenuation rate. "
                "~50% of values are complex, indicating significant modal losses."
            ),
            "emw.freq (GHz)": (
                "Electromagnetic eigenfrequency in GHz. Range ~28-496 GHz "
                "(microwave to sub-THz). This is the primary output / 'target' "
                "for inverse design."
            ),
        },
        "inverse_design_context": {
            "goal": (
                "Given desired electromagnetic frequency response (band structure), "
                "determine the optimal geometric parameters (L1, L2) and potentially "
                "the unit cell topology."
            ),
            "forward_problem": "Geometry (L1, L2) + k → Eigenfrequencies (freq)",
            "inverse_problem": "Desired freq/band-structure → Optimal geometry (L1, L2)",
            "ml_formulation": (
                "This is fundamentally a REGRESSION problem for the forward model, "
                "and an INVERSE DESIGN / GENERATIVE problem for the reverse mapping. "
                "The data enables training a surrogate model that replaces expensive "
                "COMSOL simulations."
            ),
        },
        "key_observations": [
            (
                "L1 has 8 values (140-189 nm, step ~7 nm). "
                "This provides a moderate parametric sweep over the first geometric dimension."
            ),
            (
                "L2 has 8 discrete values (91-140 nm, step ~7 nm). Combined with L1, "
                "this gives 64 (L1,L2) design configurations in the sweep."
            ),
            (
                "k sweeps 281 values (0.1 to 2.9). This provides dense band-structure "
                "coverage along the Brillouin zone."
            ),
            (
                "~50% of eigenvalues (lambda) are complex, indicating lossy/evanescent modes. "
                "These modes have physical significance (e.g., radiation losses, material absorption)."
            ),
            (
                "The frequency range 28-496 GHz spans microwave to sub-THz. "
                "This is consistent with mm-wave photonic crystal or metamaterial design."
            ),
            (
                "Two separate files (para.csv, r1.csv) with different meshes but same "
                "parameter space suggests two different unit cell geometries being compared."
            ),
            (
                "~29.5M and ~31.9M data lines respectively. Each line is a mesh node "
                "with its field solution. The mesh has ~tens of thousands of nodes per "
                "eigenmode."
            ),
        ],
        "data_challenges": [
            "Massive dataset (~8 GB total) requires out-of-core processing",
            "Spatial mesh data (X,Y per node) — need to aggregate to per-eigenmode features",
            "Complex eigenvalues need special handling (real/imag decomposition)",
            "64 (L1,L2) combinations — moderate design space, may benefit from more points",
            "8 values per parameter — interpolation viable but extrapolation risky",
            "Node-level data is redundant for band-structure analysis (need mode-level aggregation)",
        ],
    }


# ============================================================
# Recommendations for ML/inverse design
# ============================================================

def ml_recommendations():
    """Generate ML and inverse design recommendations."""
    return {
        "data_preprocessing_roadmap": {
            "step_1_aggregation": {
                "description": (
                    "Aggregate node-level data to eigenmode-level. Each unique "
                    "(L1, L2, k, freq) combination should become ONE row. "
                    "The spatial (X, Y) data should be aggregated into features."
                ),
                "output_columns": [
                    "L1", "L2", "k", "freq_ghz",
                    "lambda_real", "lambda_imag", "lambda_magnitude",
                    "is_lossy (bool)", "mesh_node_count",
                ],
                "expected_rows": "~thousands (much smaller than 30M)",
            },
            "step_2_band_structure": {
                "description": (
                    "For each (L1, L2), build the photonic band structure: "
                    "freq vs k curves for each band index."
                ),
                "output": "band_structure[L2][band_index] = array of (k, freq) pairs",
            },
            "step_3_feature_engineering": {
                "suggested_features": [
                    "Band gaps (frequency ranges with no eigenmode)",
                    "Band widths",
                    "Group velocity (d(freq)/dk)",
                    "Mode loss (imaginary part of lambda)",
                    "Band gap to midgap ratio",
                    "Number of modes per frequency window",
                    "Density of states",
                ],
            },
            "step_4_normalization": {
                "description": "Normalize frequencies and k-values for ML training",
                "suggestions": [
                    "Normalize freq by c/(2*L1) for dimensionless frequency",
                    "Normalize k to [0, 1] or [0, π/a]",
                ],
            },
        },
        "ml_approaches": {
            "forward_surrogate": {
                "goal": "Predict eigenfrequencies from (L2, k)",
                "models": [
                    "Neural Network (MLP) — baseline",
                    "Physics-Informed Neural Network (PINN)",
                    "Gaussian Process Regression — good for small data (3 L2 values)",
                    "Random Forest / XGBoost — for tabular baseline",
                ],
                "metrics": ["MAE (GHz)", "RMSE (GHz)", "R²", "Max error"],
            },
            "inverse_design": {
                "goal": "Given desired band structure → predict L2 (and future: L1, topology)",
                "approaches": [
                    "Tandem network (forward + inverse)",
                    "Conditional VAE",
                    "Conditional GAN (cGAN)",
                    "Bayesian optimization over forward surrogate",
                    "Reinforcement learning for topology optimization",
                ],
            },
            "data_augmentation": {
                "note": (
                    "With 64 (L1,L2) combinations and ~7 nm step, the design space has moderate "
                    "coverage. Consider: (1) running more COMSOL sweeps with finer grid, "
                    "(2) interpolation-based augmentation, (3) transfer learning from "
                    "analytical models to improve inter-point predictions."
                ),
            },
        },
        "critical_questions_for_roadmap": [
            "What is the target application? (e.g., 5G filter, THz absorber, waveguide)",
            "What specific frequency response is desired? (band gap location, bandwidth)",
            "Will L1 also be varied in future simulations?",
            "Can more L2 values be simulated to densify the design space?",
            "Are there manufacturing constraints on L1 and L2?",
            "Is the 2D simulation sufficient or is 3D needed?",
            "What is the acceptable error tolerance for the surrogate model?",
            "Is real-time inference needed (online design tool) or batch optimization?",
            "Are there other geometric parameters (e.g., hole shape, material) to explore?",
            "What is the relationship between para.csv and r1.csv geometries?",
        ],
    }


# ============================================================
# Report generation
# ============================================================

def generate_markdown_report(report):
    """Generate the LLM-ready Markdown report."""
    r = report
    para = r["datasets"]["para"]
    r1 = r["datasets"]["r1"]
    cross = r["cross_file_analysis"]
    physics = r["physics_interpretation"]
    ml = r["ml_recommendations"]

    lines = []
    lines.append("# COMPREHENSIVE DATASET INTELLIGENCE REPORT")
    lines.append("")
    lines.append("> **Purpose**: Complete analysis of two COMSOL electromagnetic simulation")
    lines.append("> datasets for inverse design of photonic/metamaterial unit cells.")
    lines.append("> This report is designed to be given to an LLM (Claude) to generate")
    lines.append("> a complete project roadmap.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- Section 1: Executive Summary ----
    lines.append("## 1. EXECUTIVE SUMMARY")
    lines.append("")
    lines.append("| Property | para.csv | r1.csv |")
    lines.append("|----------|----------|--------|")
    lines.append(f"| File size | {para['full_pass']['file_size_gb']} GB | {r1['full_pass']['file_size_gb']} GB |")
    lines.append(f"| Data lines | {para['full_pass']['data_lines']:,} | {r1['full_pass']['data_lines']:,} |")
    lines.append(f"| Columns | {len(para['full_pass']['columns'])} | {len(r1['full_pass']['columns'])} |")
    lines.append(f"| L1 values | {para['full_pass']['parametric_sweep']['L1']['count']} | {r1['full_pass']['parametric_sweep']['L1']['count']} |")
    lines.append(f"| L2 values | {para['full_pass']['parametric_sweep']['L2']['count']} | {r1['full_pass']['parametric_sweep']['L2']['count']} |")
    lines.append(f"| k values | {para['full_pass']['parametric_sweep']['k']['count']} | {r1['full_pass']['parametric_sweep']['k']['count']} |")
    lines.append(f"| Freq range (GHz) | {para['full_pass']['frequency_output']['min_ghz']}-{para['full_pass']['frequency_output']['max_ghz']} | {r1['full_pass']['frequency_output']['min_ghz']}-{r1['full_pass']['frequency_output']['max_ghz']} |")
    lines.append(f"| Complex λ fraction | {para['full_pass']['lambda_column']['complex_fraction']:.1%} | {r1['full_pass']['lambda_column']['complex_fraction']:.1%} |")
    lines.append("")

    # ---- Section 2: Domain & Physics ----
    lines.append("## 2. DOMAIN & PHYSICS CONTEXT")
    lines.append("")
    lines.append(f"**Domain**: {physics['domain']}")
    lines.append(f"")
    lines.append(f"**Software**: {physics['simulation_software']}")
    lines.append(f"")
    lines.append(f"**Problem Type**: {physics['problem_type']}")
    lines.append(f"")
    lines.append(f"**Context**: {physics['context']}")
    lines.append("")
    lines.append("### Column Definitions")
    lines.append("")
    lines.append("| Column | Description |")
    lines.append("|--------|-------------|")
    for col, desc in physics["columns_explained"].items():
        lines.append(f"| `{col}` | {desc} |")
    lines.append("")

    # ---- Section 3: Inverse Design ----
    lines.append("## 3. INVERSE DESIGN CONTEXT")
    lines.append("")
    for key, val in physics["inverse_design_context"].items():
        lines.append(f"**{key.replace('_', ' ').title()}**: {val}")
        lines.append("")

    # ---- Section 4: Key Observations ----
    lines.append("## 4. KEY OBSERVATIONS")
    lines.append("")
    for i, obs in enumerate(physics["key_observations"], 1):
        lines.append(f"{i}. {obs}")
        lines.append("")

    # ---- Section 5: Parametric Sweep Details ----
    lines.append("## 5. PARAMETRIC SWEEP DETAILS")
    lines.append("")
    for fname, data in [("para.csv", para), ("r1.csv", r1)]:
        fp = data["full_pass"]
        lines.append(f"### {fname}")
        lines.append("")
        ps = fp["parametric_sweep"]
        lines.append(f"- **L1**: {ps['L1']['count']} value(s) → {ps['L1']['values_nm']} nm")
        lines.append(f"- **L2**: {ps['L2']['count']} value(s) → {ps['L2']['values_nm']} nm")
        lines.append(f"- **k**: {ps['k']['count']} values, range [{ps['k']['min']}, {ps['k']['max']}], step {ps['k']['step']}")
        lines.append(f"- **Frequencies**: {fp['frequency_output']['unique_count']} unique values, range [{fp['frequency_output']['min_ghz']}, {fp['frequency_output']['max_ghz']}] GHz")
        lines.append(f"- **λ complex fraction**: {fp['lambda_column']['complex_fraction']:.1%}")
        lines.append("")

    # ---- Section 6: Statistical Summary ----
    lines.append("## 6. STATISTICAL SUMMARY (SAMPLED)")
    lines.append("")
    for fname, data in [("para.csv", para), ("r1.csv", r1)]:
        stats = data["statistics"]
        lines.append(f"### {fname} (sample size: {stats['sample_size']:,})")
        lines.append("")
        lines.append("| Column | Min | Max | Mean | Std | Median | Skew | Zeros |")
        lines.append("|--------|-----|-----|------|-----|--------|------|-------|")
        for col_key, cs in stats["column_statistics"].items():
            if cs.get("count", 0) == 0:
                continue
            lines.append(
                f"| {cs['name']} | {cs.get('min', 'N/A')} | {cs.get('max', 'N/A')} "
                f"| {cs.get('mean', 'N/A')} | {cs.get('std', 'N/A')} "
                f"| {cs.get('median', 'N/A')} | {cs.get('skewness', 'N/A')} "
                f"| {cs.get('zero_count', 0)} |"
            )
        lines.append("")

    # ---- Section 7: Frequency Distribution ----
    lines.append("## 7. FREQUENCY DISTRIBUTION (10 GHz BINS)")
    lines.append("")
    for fname, data in [("para.csv", para), ("r1.csv", r1)]:
        stats = data["statistics"]
        lines.append(f"### {fname}")
        lines.append("")
        dist = stats.get("frequency_distribution_10GHz_bins", {})
        if dist:
            lines.append("| Frequency Band | Count |")
            lines.append("|----------------|-------|")
            for band, count in sorted(dist.items(), key=lambda x: int(x[0].split("-")[0])):
                lines.append(f"| {band} | {count:,} |")
        lines.append("")

    # ---- Section 8: Correlations ----
    lines.append("## 8. CORRELATIONS")
    lines.append("")
    for fname, data in [("para.csv", para), ("r1.csv", r1)]:
        stats = data["statistics"]
        corr = stats.get("correlations", {})
        lines.append(f"### {fname}")
        lines.append(f"- freq ↔ k correlation: `{corr.get('freq_vs_k', 'N/A')}`")
        lines.append(f"- freq ↔ |λ| correlation: `{corr.get('freq_vs_lambda_magnitude', 'N/A')}`")
        lines.append("")

    # ---- Section 9: Eigenmode Analysis ----
    lines.append("## 9. EIGENMODE / BAND ANALYSIS")
    lines.append("")
    for fname, data in [("para.csv", para), ("r1.csv", r1)]:
        stats = data["statistics"]
        ea = stats.get("eigenmode_analysis", {})
        lines.append(f"### {fname}")
        lines.append(f"- Avg bands per k-point: {ea.get('avg_bands_per_k', 'N/A')}")
        lines.append(f"- Min bands per k-point: {ea.get('min_bands_per_k', 'N/A')}")
        lines.append(f"- Max bands per k-point: {ea.get('max_bands_per_k', 'N/A')}")
        lines.append(f"- {ea.get('interpretation', '')}")
        lines.append("")

    # ---- Section 10: Cross-file Analysis ----
    lines.append("## 10. CROSS-FILE COMPARISON")
    lines.append("")
    lines.append(f"- **Identical schema**: {cross['schema_comparison']['identical_schema']}")
    lines.append(f"- **Row ratio (r1/para)**: {cross['size_comparison']['ratio_r1_to_para']}")
    lines.append(f"- **Same L1**: {cross['parameter_comparison']['L1_same']}")
    lines.append(f"- **Same L2**: {cross['parameter_comparison']['L2_same']}")
    lines.append(f"- **Same k range**: {cross['parameter_comparison']['k_range_same']}")
    lines.append("")
    lines.append("### Geometry Differences")
    lines.append(f"- para.csv X range: {cross['geometry_comparison']['para_x_range']}")
    lines.append(f"- r1.csv X range: {cross['geometry_comparison']['r1_x_range']}")
    lines.append(f"- para.csv Y range: {cross['geometry_comparison']['para_y_range']}")
    lines.append(f"- r1.csv Y range: {cross['geometry_comparison']['r1_y_range']}")
    lines.append(f"- para.csv unique XY (5M): {cross['geometry_comparison']['para_unique_xy_5M']}")
    lines.append(f"- r1.csv unique XY (5M): {cross['geometry_comparison']['r1_unique_xy_5M']}")
    lines.append("")
    lines.append(f"**Interpretation**: {cross['geometry_comparison']['interpretation']}")
    lines.append("")
    lines.append("### Likely Relationship")
    lines.append(f"{cross['likely_relationship']['hypothesis']}")
    lines.append("")

    # ---- Section 11: Data Quality ----
    lines.append("## 11. DATA QUALITY")
    lines.append("")
    for fname, data in [("para.csv", para), ("r1.csv", r1)]:
        dq = data["data_quality"]
        lines.append(f"### {fname}")
        lines.append(f"- Lines checked: {dq['lines_checked']:,}")
        lines.append(f"- Parse errors: {dq['parse_errors']}")
        lines.append(f"- Short lines: {dq['short_lines_under_7_fields']}")
        lines.append(f"- NaN frequencies: {dq['nan_frequency']}")
        lines.append(f"- Negative frequencies: {dq['negative_frequency']}")
        lines.append(f"- Adjacent duplicates: {dq['adjacent_duplicate_lines']}")
        lines.append(f"- **Assessment: {dq['quality_assessment']}**")
        lines.append("")

    # ---- Section 12: Data Challenges ----
    lines.append("## 12. DATA CHALLENGES FOR ML")
    lines.append("")
    for i, challenge in enumerate(physics["data_challenges"], 1):
        lines.append(f"{i}. {challenge}")
    lines.append("")

    # ---- Section 13: ML Recommendations ----
    lines.append("## 13. ML / INVERSE DESIGN RECOMMENDATIONS")
    lines.append("")

    prep = ml["data_preprocessing_roadmap"]
    lines.append("### Preprocessing Roadmap")
    lines.append("")
    for step_key, step in prep.items():
        title = step_key.replace("_", " ").title()
        lines.append(f"#### {title}")
        lines.append(f"{step.get('description', '')}")
        if "output_columns" in step:
            lines.append(f"- Output columns: `{step['output_columns']}`")
        if "expected_rows" in step:
            lines.append(f"- Expected rows: {step['expected_rows']}")
        if "suggested_features" in step:
            for feat in step["suggested_features"]:
                lines.append(f"  - {feat}")
        if "output" in step:
            lines.append(f"- Output: `{step['output']}`")
        if "suggestions" in step:
            for s in step["suggestions"]:
                lines.append(f"  - {s}")
        lines.append("")

    lines.append("### ML Approaches")
    lines.append("")
    for approach_key, approach in ml["ml_approaches"].items():
        title = approach_key.replace("_", " ").title()
        lines.append(f"#### {title}")
        lines.append(f"**Goal**: {approach.get('goal', approach.get('note', ''))}")
        if "models" in approach:
            lines.append("**Models**:")
            for m in approach["models"]:
                lines.append(f"  - {m}")
        if "approaches" in approach:
            lines.append("**Approaches**:")
            for a in approach["approaches"]:
                lines.append(f"  - {a}")
        if "metrics" in approach:
            lines.append(f"**Metrics**: {approach['metrics']}")
        lines.append("")

    # ---- Section 14: Critical Questions ----
    lines.append("## 14. CRITICAL QUESTIONS BEFORE BUILDING ROADMAP")
    lines.append("")
    for i, q in enumerate(ml["critical_questions_for_roadmap"], 1):
        lines.append(f"{i}. {q}")
    lines.append("")

    # ---- Section 15: Raw JSON ----
    lines.append("---")
    lines.append("")
    lines.append("## APPENDIX: RAW STRUCTURED DATA (JSON)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(r, indent=2, default=str, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("DATASET DETAILS REPORT GENERATOR")
    print("COMSOL EMW Eigenfrequency Simulation Data")
    print("=" * 70)
    print()

    # Check files exist
    for name, path in FILES.items():
        if not path.exists():
            print(f"ERROR: {path} not found!")
            sys.exit(1)
        print(f"  Found: {path.name} ({os.path.getsize(path)/1024**3:.2f} GB)")

    print()

    # ---- Full pass analysis ----
    print("PHASE 1: Full-pass parameter discovery and line counting")
    print("-" * 50)
    para_full = full_pass_analysis(FILES["para"], "para")
    r1_full = full_pass_analysis(FILES["r1"], "r1")
    print()

    # ---- Statistical sampling ----
    print("PHASE 2: Statistical sampling")
    print("-" * 50)
    para_stats = statistical_analysis(FILES["para"], "para", STAT_SAMPLE_SIZE)
    r1_stats = statistical_analysis(FILES["r1"], "r1", STAT_SAMPLE_SIZE)
    print()

    # ---- Data quality ----
    print("PHASE 3: Data quality checks")
    print("-" * 50)
    para_quality = data_quality_analysis(FILES["para"], "para")
    r1_quality = data_quality_analysis(FILES["r1"], "r1")
    print()

    # ---- Cross-file ----
    print("PHASE 4: Cross-file comparison")
    print("-" * 50)
    cross = cross_file_analysis(para_full, r1_full)
    print()

    # ---- Physics interpretation ----
    print("PHASE 5: Physics and ML interpretation")
    print("-" * 50)
    physics = physics_interpretation(para_full, r1_full, para_stats, r1_stats)
    ml = ml_recommendations()
    print("  Done")
    print()

    # ---- Build final report ----
    report = {
        "report_metadata": {
            "title": "Comprehensive Dataset Intelligence Report",
            "files": [str(p) for p in FILES.values()],
            "domain": "Computational Electromagnetics / Inverse Design",
            "generated_by": "details.py",
        },
        "datasets": {
            "para": {
                "full_pass": para_full,
                "statistics": para_stats,
                "data_quality": para_quality,
            },
            "r1": {
                "full_pass": r1_full,
                "statistics": r1_stats,
                "data_quality": r1_quality,
            },
        },
        "cross_file_analysis": cross,
        "physics_interpretation": physics,
        "ml_recommendations": ml,
    }

    # ---- Save ----
    print("PHASE 6: Generating reports")
    print("-" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_content = generate_markdown_report(report)
    OUTPUT_MD.write_text(md_content, encoding="utf-8")
    print(f"  Markdown: {OUTPUT_MD}")

    # JSON
    json_content = json.dumps(report, indent=2, default=str, ensure_ascii=False)
    OUTPUT_JSON.write_text(json_content, encoding="utf-8")
    print(f"  JSON:     {OUTPUT_JSON}")

    print()
    print("=" * 70)
    print("REPORT COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Open  reports/dataset_details_report.md")
    print("  2. Copy its contents into Claude")
    print("  3. Ask Claude to generate the inverse design roadmap")
    print()


if __name__ == "__main__":
    main()
