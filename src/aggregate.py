import os
import sys
import time
import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COLS = ["X", "Y", "L1", "L2", "k", "lam", "f_ghz"]
NUM = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
LAM_RE = rf"^({NUM})(?:([+-]?{NUM})i)?$"

def aggregate_file(csv_path: str, structure_id: int, structure_name: str) -> pl.DataFrame:
    print(f"[{time.strftime('%H:%M:%S')}] Starting aggregation of {csv_path} (Structure {structure_id}: {structure_name})...")
    t0 = time.time()
    
    # 1. Group by mode parameters to compress ~30M rows into ~36K eigenmodes
    lf = (pl.scan_csv(csv_path, comment_prefix="%", has_header=False, new_columns=COLS,
                      schema_overrides={"lam": pl.Utf8})
          .group_by(["L1", "L2", "k", "lam", "f_ghz"])
          .agg(pl.len().alias("n_nodes")))
    
    print(f"[{time.strftime('%H:%M:%S')}] Executing streaming aggregation plan...")
    df = lf.collect(engine="streaming")
    t1 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Aggregated {csv_path} in {t1 - t0:.1f}s. Extracted {len(df)} eigenmode records.")
    
    # 2. Extract real/imag components, grid indices, and sort
    df = (df.with_columns(
            lam_re=pl.col("lam").str.extract(LAM_RE, 1).cast(pl.Float64),
            lam_im=pl.col("lam").str.extract(LAM_RE, 2).cast(pl.Float64).fill_null(0.0),
            structure=pl.lit(structure_id, dtype=pl.Int8),
            structure_name=pl.lit(structure_name, dtype=pl.Utf8),
            # Convert raw L (meters) -> um, snap to nominal 7 um grid
            L1_um=(pl.col("L1") * 1e6).round(3),
            L2_um=(pl.col("L2") * 1e6).round(3),
            i1=((pl.col("L1") * 1e6 - 140) / 7).round().cast(pl.Int16),
            i2=((pl.col("L2") * 1e6 - 91) / 7).round().cast(pl.Int16),
            ik=(pl.col("k") * 100).round().cast(pl.Int16) - 10,
            f_round=pl.col("f_ghz").round(4)
          )
          # Deduplicate duplicate mesh domain records of the exact same physical eigenmode
          .group_by(["structure", "structure_name", "i1", "i2", "ik", "L1_um", "L2_um", "f_round"])
          .agg(
              pl.col("f_ghz").first(),
              pl.col("lam_re").first(),
              pl.col("lam_im").mean(),
              pl.col("n_nodes").sum()
          )
          .sort(["structure", "i1", "i2", "ik", "f_ghz"])
          .with_columns(
              band=pl.col("f_ghz").cum_count().over(["structure", "i1", "i2", "ik"]).cast(pl.Int8)
          )
          .drop("f_round"))
    
    return df

def build_tensor(modes_df: pl.DataFrame, n_bands: int = 4) -> np.ndarray:
    """F[structure, i1, i2, ik, band_idx] in GHz, NaN where band is missing."""
    F = np.full((2, 8, 8, 281, n_bands), np.nan, dtype=np.float32)
    sub = modes_df.filter(pl.col("band") <= n_bands)
    s = sub["structure"].to_numpy()
    i1 = sub["i1"].to_numpy()
    i2 = sub["i2"].to_numpy()
    ik = sub["ik"].to_numpy()
    b = sub["band"].to_numpy() - 1
    f = sub["f_ghz"].to_numpy()
    F[s, i1, i2, ik, b] = f
    return F

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(root)
    
    para_path = os.path.join(workspace_root, "para.csv")
    r1_path = os.path.join(workspace_root, "r1.csv")
    
    interim_dir = os.path.join(root, "data", "interim")
    processed_dir = os.path.join(root, "data", "processed")
    docs_dir = os.path.join(root, "docs")
    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    # 1. Process para.csv
    df_para = aggregate_file(para_path, structure_id=0, structure_name="para")
    para_parquet = os.path.join(interim_dir, "modes_para.parquet")
    df_para.write_parquet(para_parquet)
    print(f"Saved {para_parquet} ({os.path.getsize(para_parquet) / 1024 / 1024:.2f} MB)")
    
    # 2. Process r1.csv
    df_r1 = aggregate_file(r1_path, structure_id=1, structure_name="r1")
    r1_parquet = os.path.join(interim_dir, "modes_r1.parquet")
    df_r1.write_parquet(r1_parquet)
    print(f"Saved {r1_parquet} ({os.path.getsize(r1_parquet) / 1024 / 1024:.2f} MB)")
    
    # 3. Combine both
    df_all = pl.concat([df_para, df_r1])
    all_parquet = os.path.join(processed_dir, "modes_all.parquet")
    df_all.write_parquet(all_parquet)
    print(f"Saved combined {all_parquet} ({os.path.getsize(all_parquet) / 1024 / 1024:.2f} MB)")
    
    # 4. Construct band tensor F (2, 8, 8, 281, 4)
    F = build_tensor(df_all, n_bands=4)
    tensor_path = os.path.join(processed_dir, "F.npy")
    np.save(tensor_path, F)
    print(f"Saved band tensor {tensor_path}, shape: {F.shape}")
    
    # 5. G0 Gate validation checks
    print("\n--- Gate G0 Verification Checks ---")
    for name, df in [("para", df_para), ("r1", df_r1)]:
        pts = df.select(["i1", "i2", "ik"]).unique()
        n_pts = len(pts)
        expected_pts = 8 * 8 * 281  # 17,984
        print(f"[{name}] Unique (design, k) points: {n_pts} / {expected_pts} (Match: {n_pts == expected_pts})")
        
        # Mode count distribution per (i1, i2, ik)
        mode_counts = (df.group_by(["i1", "i2", "ik"])
                         .agg(pl.len().alias("modes_per_point"))
                         .group_by("modes_per_point")
                         .agg(pl.len().alias("point_count"))
                         .sort("modes_per_point"))
        print(f"[{name}] Mode count breakdown per point:")
        for row in mode_counts.iter_rows(named=True):
            print(f"   {row['modes_per_point']} modes: {row['point_count']} points ({row['point_count']/n_pts*100:.2f}%)")
        
        # Re(lambda) vs f_ghz verification
        diff = (df["lam_re"] / 1e9 - df["f_ghz"]).abs()
        max_diff = diff.max()
        print(f"[{name}] Max |Re(lambda)/1e9 - f_ghz| = {max_diff:.3e}")
        
        # Complex lambda stats
        comp_count = (df["lam_im"].abs() > 1e-3).sum()
        print(f"[{name}] Non-zero Im(lambda) count at mode level: {comp_count} / {len(df)} ({comp_count/len(df)*100:.2f}%)")
        print(f"[{name}] Max Im(lambda) = {df['lam_im'].abs().max():.3e}, Median Im(lambda) = {df['lam_im'].median()}")

    print("\nAggregation & G0 checks complete!")

if __name__ == "__main__":
    main()
