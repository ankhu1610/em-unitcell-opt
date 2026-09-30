"""
dataset_report.py

Generate ONE LLM-ready intelligence report from:
    - para.csv
    - r1.csv

Both CSV files must be in the same directory as this script.

Run:
    python dataset_report.py

Jupyter / VS Code:
    %run dataset_report.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

# Exact datasets for this project.
# Keep these CSV files in the same directory as this script.
BASE_DIR = Path(__file__).resolve().parent

DATASET_FILES = [
   
    BASE_DIR / "r1.csv",
]

OUTPUT_DIR = BASE_DIR / "reports"
DEFAULT_OUTPUT = OUTPUT_DIR / "dataset_intelligence.md"

MAX_UNIQUE_VALUES = 30
TOP_VALUES = 10
SAMPLE_ROWS = 5
MAX_CORRELATIONS = 50
MAX_COLUMNS_IN_PREVIEW = 500


# ============================================================
# Utility
# ============================================================

def safe_json(obj):
    """Convert pandas/numpy objects to JSON-safe values."""
    if obj is None:
        return None

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.Timestamp):
        return str(obj)

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def safe_preview(df):
    """Return a small JSON-safe preview."""
    preview = df.head(SAMPLE_ROWS)

    if len(preview.columns) > MAX_COLUMNS_IN_PREVIEW:
        preview = preview.iloc[:, :MAX_COLUMNS_IN_PREVIEW]

    return preview.to_dict(orient="records")


# ============================================================
# Dataset Discovery
# ============================================================

def discover_datasets(data_dir):
    """Recursively find every supported dataset file."""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir.absolute()}"
        )

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but received: {data_dir}"
        )

    files = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(path):
    """Load one supported dataset."""
    path = Path(path)
    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in {".xlsx", ".xls"}:
        # For workbooks, read the first sheet.
        # Sheet information is separately captured below.
        return pd.read_excel(path)

    if extension == ".parquet":
        return pd.read_parquet(path)

    if extension == ".json":
        return pd.read_json(path)

    raise ValueError(f"Unsupported extension: {extension}")


def workbook_sheets(path):
    """Return Excel sheet names without loading every sheet."""
    if Path(path).suffix.lower() not in {".xlsx", ".xls"}:
        return []

    try:
        return pd.ExcelFile(path).sheet_names
    except Exception:
        return []


# ============================================================
# Column Inference
# ============================================================

def infer_column_role(series):
    name = str(series.name).lower()

    if any(x in name for x in ["id", "uuid", "identifier", "key", "index"]):
        return "identifier_candidate"

    if any(
        x in name
        for x in ["date", "time", "timestamp", "created", "updated"]
    ):
        return "datetime_candidate"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique(dropna=True)

        if nunique <= 10:
            return "low_cardinality_numeric"

        return "continuous_numeric"

    if pd.api.types.is_string_dtype(series):
        avg_length = series.dropna().astype(str).str.len().mean()

        if pd.notna(avg_length) and avg_length > 50:
            return "text_candidate"

        return "categorical_or_text"

    return "unknown"


# ============================================================
# Dataset Overview
# ============================================================

def dataset_overview(df, path):
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    return {
        "file_name": path.name,
        "file_path": str(path.absolute()),
        "file_extension": path.suffix.lower(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(memory_mb, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percentage": round(
            float(df.duplicated().mean() * 100), 3
        ),
    }


# ============================================================
# Column Analysis
# ============================================================

def analyze_columns(df):
    results = []

    for col in df.columns:
        s = df[col]

        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))

        info = {
            "column": str(col),
            "dtype": str(s.dtype),
            "role": infer_column_role(s),
            "missing_count": missing,
            "missing_percentage": round(
                float(s.isna().mean() * 100), 3
            ),
            "unique_values": unique,
            "unique_percentage": round(
                float(unique / max(len(df), 1) * 100), 3
            ),
        }

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()

            info.update({
                "mean": safe_json(desc.get("mean")),
                "std": safe_json(desc.get("std")),
                "min": safe_json(desc.get("min")),
                "25_percentile": safe_json(desc.get("25%")),
                "median": safe_json(desc.get("50%")),
                "75_percentile": safe_json(desc.get("75%")),
                "max": safe_json(desc.get("max")),
                "skewness": safe_json(s.skew()),
                "zero_count": int((s == 0).sum()),
            })

        else:
            top_values = (
                s.value_counts(dropna=False)
                .head(TOP_VALUES)
                .to_dict()
            )

            info["top_values"] = {
                str(k): safe_json(v)
                for k, v in top_values.items()
            }

            if unique <= MAX_UNIQUE_VALUES:
                info["all_values"] = [
                    str(x) for x in s.dropna().unique()
                ]

        results.append(info)

    return results


# ============================================================
# Missing Values
# ============================================================

def missing_analysis(df):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    return [
        {
            "column": str(col),
            "missing_count": int(count),
            "missing_percentage": round(
                float(count / max(len(df), 1) * 100), 3
            ),
        }
        for col, count in missing.items()
    ]


# ============================================================
# Correlation
# ============================================================

def correlation_analysis(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:
        return []

    corr = numeric.corr()
    result = []

    columns = corr.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]

            if pd.notna(value):
                result.append({
                    "feature_1": str(columns[i]),
                    "feature_2": str(columns[j]),
                    "correlation": round(float(value), 4),
                })

    result.sort(
        key=lambda x: abs(x["correlation"]),
        reverse=True,
    )

    return result[:MAX_CORRELATIONS]


# ============================================================
# Outliers
# ============================================================

def outlier_analysis(df):
    results = []

    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()

        if len(s) < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((s < lower) | (s > upper)).sum()

        results.append({
            "column": str(col),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "outlier_count": int(outliers),
            "outlier_percentage": round(
                float(outliers / len(s) * 100), 3
            ),
        })

    results.sort(
        key=lambda x: x["outlier_percentage"],
        reverse=True,
    )

    return results


# ============================================================
# Identifier / Target Candidates
# ============================================================

def detect_identifier_columns(df):
    identifiers = []

    for col in df.columns:
        s = df[col]
        unique_ratio = s.nunique(dropna=True) / max(len(df), 1)
        name = str(col).lower()

        if (
            unique_ratio > 0.95
            or any(
                x in name
                for x in ["id", "uuid", "identifier", "key"]
            )
        ):
            identifiers.append({
                "column": str(col),
                "unique_ratio": round(float(unique_ratio), 4),
            })

    return identifiers


def detect_target_candidates(df):
    candidates = []

    target_words = [
        "target",
        "label",
        "class",
        "outcome",
        "target_value",
        "price",
        "sales",
        "score",
        "churn",
        "fraud",
        "y",
    ]

    for col in df.columns:
        name = str(col).lower()
        score = 0
        reasons = []

        if any(word in name for word in target_words):
            score += 2
            reasons.append("column name may indicate target")

        unique = df[col].nunique(dropna=True)

        if unique == 2:
            score += 1
            reasons.append("binary cardinality")

        elif 2 < unique < 20:
            score += 0.5
            reasons.append("low cardinality")

        if score > 0:
            candidates.append({
                "column": str(col),
                "score": score,
                "reasons": reasons,
            })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates


# ============================================================
# Datetime
# ============================================================

def datetime_analysis(df):
    results = []

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            parsed = s.dropna()

        elif pd.api.types.is_object_dtype(s):
            name = str(col).lower()

            if not any(
                x in name
                for x in ["date", "time", "timestamp"]
            ):
                continue

            converted = pd.to_datetime(
                s,
                errors="coerce",
            )

            if converted.notna().mean() < 0.8:
                continue

            parsed = converted.dropna()

        else:
            continue

        if len(parsed) == 0:
            continue

        results.append({
            "column": str(col),
            "min": str(parsed.min()),
            "max": str(parsed.max()),
            "range_days": int(
                (parsed.max() - parsed.min()).days
            ),
        })

    return results


# ============================================================
# Data Quality
# ============================================================

def data_quality_checks(df):
    checks = []

    constant_columns = [
        str(col)
        for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    checks.append({
        "check": "constant_columns",
        "count": len(constant_columns),
        "columns": constant_columns,
    })

    high_cardinality = []

    for col in df.columns:
        ratio = df[col].nunique(dropna=True) / max(len(df), 1)

        if ratio > 0.8:
            high_cardinality.append({
                "column": str(col),
                "unique_ratio": round(float(ratio), 4),
            })

    checks.append({
        "check": "high_cardinality_columns",
        "count": len(high_cardinality),
        "columns": high_cardinality,
    })

    duplicate_columns = []
    columns = list(df.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if df[columns[i]].equals(df[columns[j]]):
                duplicate_columns.append(
                    [str(columns[i]), str(columns[j])]
                )

    checks.append({
        "check": "duplicate_columns",
        "count": len(duplicate_columns),
        "columns": duplicate_columns,
    })

    return checks


# ============================================================
# Single Dataset Report
# ============================================================

def analyze_dataset(path):
    """Analyze one file and return a structured report."""
    df = load_dataset(path)

    report = {
        "file": dataset_overview(df, path),
        "excel_sheets": workbook_sheets(path),
        "columns": analyze_columns(df),
        "missing_values": missing_analysis(df),
        "correlations": correlation_analysis(df),
        "outliers": outlier_analysis(df),
        "identifier_candidates": detect_identifier_columns(df),
        "target_candidates": detect_target_candidates(df),
        "datetime_analysis": datetime_analysis(df),
        "data_quality_checks": data_quality_checks(df),
        "sample_rows": safe_preview(df),
    }

    return report


# ============================================================
# Cross-Dataset Analysis
# ============================================================

def cross_dataset_analysis(dataset_reports):
    """
    Compare schemas across all discovered datasets.
    Useful for train/test/validation or multi-table projects.
    """
    result = {
        "dataset_count": len(dataset_reports),
        "datasets": [],
        "common_columns": [],
        "all_columns": {},
        "potential_train_test_pairs": [],
    }

    column_sets = []

    for report in dataset_reports:
        file_name = report["file"]["file_name"]
        columns = [
            c["column"]
            for c in report["columns"]
        ]

        column_sets.append(set(columns))

        result["datasets"].append({
            "file": file_name,
            "rows": report["file"]["rows"],
            "columns": report["file"]["columns"],
            "column_names": columns,
        })

        for column in columns:
            result["all_columns"].setdefault(
                column, []
            ).append(file_name)

    if column_sets:
        common = set.intersection(*column_sets)

        result["common_columns"] = sorted(common)

    # Detect likely train/test/validation relationships.
    names = [
        report["file"]["file_name"].lower()
        for report in dataset_reports
    ]

    train_files = [
        n for n in names
        if "train" in n
    ]

    test_files = [
        n for n in names
        if "test" in n
    ]

    validation_files = [
        n for n in names
        if any(x in n for x in ["valid", "validation", "val"])
    ]

    result["potential_train_test_pairs"] = {
        "train": train_files,
        "test": test_files,
        "validation": validation_files,
    }

    return result


# ============================================================
# LLM Prompt
# ============================================================

def generate_llm_prompt(report):
    report_json = json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""# DATASET INTELLIGENCE PACKAGE FOR AN LLM

## Purpose

This document contains an automatically generated analysis of the TWO
project datasets: `para.csv` and `r1.csv`.

Use this document as the factual starting point for designing the project.
Treat `para.csv` and `r1.csv` as potentially related datasets, but do not
assume their relationship until the evidence in this report supports it.

Important:
- Do not assume facts that are not present in this report.
- Clearly distinguish observations from hypotheses.
- If the target/problem is ambiguous, say so.
- Do not blindly recommend deep learning, LLMs, RAG, agents, vector databases,
  Kubernetes, or other infrastructure.
- Choose architecture based on the actual evidence.
- Think like a production ML engineer, not only a Kaggle competitor.

---

# 1. DATASET INVENTORY

Understand:
- how many files exist
- what each file appears to represent
- which files may be train/test/validation
- whether multiple files share schemas
- whether files may represent different tables/entities

---

# 2. PROBLEM FORMULATION

Infer the most likely ML/data problem.

Consider:
- classification
- regression
- ranking
- recommendation
- forecasting
- anomaly detection
- clustering
- NLP
- computer vision
- multimodal
- other

Do not force a problem formulation if the dataset does not support one.

---

# 3. DATA UNDERSTANDING

For every dataset inspect:

- rows
- columns
- data types
- numerical features
- categorical features
- text features
- datetime features
- identifiers
- target candidates
- missing values
- duplicates
- cardinality
- distributions
- outliers
- correlations

---

# 4. RELATIONSHIP BETWEEN FILES

Determine whether files appear to be:

- train/test
- train/validation/test
- different database tables
- different versions
- independent datasets
- temporal partitions

Look for:
- common columns
- possible join keys
- schema mismatches
- distribution differences
- duplicated records

Do not invent relationships without evidence.

---

# 5. DATA QUALITY

Identify:

- missing data
- duplicate records
- duplicate columns
- constant columns
- high-cardinality columns
- suspicious values
- invalid dates
- inconsistent types
- extreme outliers

For each important issue explain its potential impact.

---

# 6. DATA LEAKAGE

Perform a serious leakage investigation.

Look for:

- target-derived features
- post-outcome features
- IDs
- timestamps
- duplicated target information
- train/test contamination
- features that would not exist at prediction time
- suspicious correlations

For every suspected leakage feature explain WHY.

---

# 7. FEATURE ENGINEERING

Create three categories:

### MUST INVESTIGATE

Features that are strongly justified by the dataset.

### SHOULD INVESTIGATE

Features that are plausible but require experiments.

### OPTIONAL

Ideas that may help but should not be implemented blindly.

For every feature engineering idea provide:

- feature
- reasoning
- expected effect
- leakage risk
- experiment required

---

# 8. BASELINE

Design the simplest credible baseline.

Specify:

- train/validation strategy
- preprocessing
- baseline model
- metrics
- cross-validation
- reproducibility requirements

Explain why this baseline is appropriate.

---

# 9. EXPERIMENT ROADMAP

Build an experiment ladder.

Example:

E0 → Data validation + baseline

E1 → Missing-value strategy

E2 → Feature engineering

E3 → Classical ML models

E4 → Advanced models

E5 → Hyperparameter optimization

E6 → Ensemble if justified

For each experiment provide:

- hypothesis
- change
- expected result
- evaluation metric
- success criterion
- what conclusion we draw from the experiment

The objective is not to run random experiments.

Every experiment should answer a question.

---

# 10. EVALUATION

Determine appropriate metrics.

Do not automatically choose accuracy.

Consider:

- primary metric
- secondary metrics
- business metric if inferable
- cross-validation
- confidence intervals
- error analysis
- subgroup performance
- calibration if relevant

---

# 11. PRODUCTION DESIGN

Design the complete production system.

Consider only components that are justified:

Data Source
    ↓
Data Ingestion
    ↓
Data Validation
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Training
    ↓
Experiment Tracking
    ↓
Model Registry
    ↓
Deployment
    ↓
Prediction API / Batch Pipeline
    ↓
Monitoring
    ↓
Feedback
    ↓
Retraining

Modify this architecture according to the actual problem.

---

# 12. MLOPS

Evaluate whether the project needs:

- Git
- Docker
- MLflow
- DVC
- FastAPI
- CI/CD
- model registry
- data validation
- feature store
- model monitoring
- data drift monitoring
- concept drift monitoring
- automated retraining

For every component explain WHY it belongs in the architecture.

---

# 13. SERVING DESIGN

If online inference is appropriate, specify:

- API design
- request schema
- response schema
- preprocessing location
- model loading
- latency considerations
- batch vs online inference
- logging

If online inference is not appropriate, explain why batch inference is preferable.

---

# 14. MONITORING

Design monitoring for:

### Data
- missing values
- schema changes
- distribution drift
- unseen categories

### Model
- prediction distribution
- performance degradation
- confidence/calibration
- drift

### System
- latency
- throughput
- failures
- resource usage

---

# 15. FAILURE MODES

Predict likely production failures:

- data drift
- concept drift
- training-serving skew
- missing values
- unseen categories
- schema changes
- leakage
- overfitting
- model degradation
- latency
- bad upstream data

For each failure:
- detection mechanism
- mitigation
- recovery strategy

---

# 16. FINAL SYSTEM ARCHITECTURE

Produce a final architecture using Mermaid.

Example:

```mermaid
flowchart TD

A[Data Sources] --> B[Data Ingestion]
B --> C[Data Validation]
C --> D[Feature Engineering]
D --> E[Training Pipeline]
E --> F[Experiment Tracking]
F --> G[Model Registry]
G --> H[Deployment]
H --> I[Prediction API]
I --> J[Monitoring]
J --> K[Feedback / Retraining]
K --> E
```

Do not blindly copy the example.

Create the architecture that fits THIS dataset.

---

# 17. IMPLEMENTATION ROADMAP

Break the project into practical phases:

Phase 1 → Understand and validate data

Phase 2 → Establish baseline

Phase 3 → Feature engineering

Phase 4 → Model experimentation

Phase 5 → Select final model

Phase 6 → Build training pipeline

Phase 7 → Build inference service

Phase 8 → Containerize

Phase 9 → Add monitoring

Phase 10 → Deploy

For every phase provide:

- objective
- files/modules to create
- expected output
- acceptance criteria

---

# 18. PROJECT DIRECTORY

Propose a production-ready project structure.

Example:

project/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── inference/
├── tests/
├── configs/
├── pipelines/
├── api/
├── monitoring/
├── Dockerfile
├── requirements.txt
└── README.md

Adapt this structure to the actual project.

---

# 19. QUESTIONS BEFORE IMPLEMENTATION

End with the most important questions that the project owner must answer before implementation.

Prioritize questions that materially affect:

- problem definition
- target
- prediction time
- data availability
- evaluation
- production constraints
- deployment
- business objective

Do not ask unnecessary questions.

---

# 20. FINAL RECOMMENDATION

End with:

1. What we know
2. What we do not know
3. What should be tested first
4. Recommended baseline
5. Recommended production architecture
6. Biggest technical risk
7. First implementation step

---

# RAW DATASET INTELLIGENCE

{report_json}
"""


# ============================================================
# Markdown Report
# ============================================================

def save_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt = generate_llm_prompt(report)

    output_path.write_text(
        prompt,
        encoding="utf-8",
    )

    # Also save raw structured JSON for programmatic use.
    json_path = output_path.with_suffix(".json")

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return output_path, json_path


# ============================================================
# Main
# ============================================================

def main(argv=None):
    """
    Analyze exactly:
        para.csv
        r1.csv

    The files must be in the same directory as this script.

    parse_known_args() is intentionally used so that this script
    also runs inside Jupyter/VS Code notebooks, where Jupyter adds
    an automatic "-f <kernel.json>" argument.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate one LLM-ready intelligence report from "
            "para.csv and r1.csv."
        )
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output Markdown report path.",
    )

    # Ignore Jupyter's automatically injected arguments.
    args, _ = parser.parse_known_args(argv)

    print("=" * 70)
    print("PARA + R1 DATASET INTELLIGENCE REPORT GENERATOR")
    print("=" * 70)

    print("\nDatasets:")

    for path in DATASET_FILES:
        print(f"  └── {path}")

    # --------------------------------------------------------
    # Verify both required files
    # --------------------------------------------------------

    missing_files = [
        path
        for path in DATASET_FILES
        if not path.exists()
    ]

    if missing_files:
        print("\nERROR: Required dataset file(s) not found:")

        for path in missing_files:
            print(f"  └── {path}")

        print(
            "\nPlace para.csv and r1.csv in the same folder "
            "as dataset_report.py."
        )

        return

    # --------------------------------------------------------
    # Analyze para.csv
    # --------------------------------------------------------

    print("\nAnalyzing para.csv...")

    try:
        para_report = analyze_dataset(DATASET_FILES[0])

        print(
            f"OK ({para_report['file']['rows']:,} rows × "
            f"{para_report['file']['columns']} columns)"
        )

    except Exception as exc:
        print(f"FAILED: para.csv -> {exc}")
        return

    # --------------------------------------------------------
    # Analyze r1.csv
    # --------------------------------------------------------

    print("\nAnalyzing r1.csv...")

    try:
        r1_report = analyze_dataset(DATASET_FILES[1])

        print(
            f"OK ({r1_report['file']['rows']:,} rows × "
            f"{r1_report['file']['columns']} columns)"
        )

    except Exception as exc:
        print(f"FAILED: r1.csv -> {exc}")
        return

    # --------------------------------------------------------
    # Cross-dataset analysis
    # --------------------------------------------------------

    print("\nComparing para.csv and r1.csv...")

    comparison = cross_dataset_analysis(
        [para_report, r1_report]
    )

    # --------------------------------------------------------
    # Build final report
    # --------------------------------------------------------

    final_report = {
        "report_metadata": {
            "report_type": "dataset_intelligence",
            "purpose": (
                "LLM-ready analysis package for para.csv and r1.csv"
            ),
            "dataset_files": [
                str(path.absolute())
                for path in DATASET_FILES
            ],
            "files_discovered": 2,
            "files_successfully_analyzed": 2,
            "files_failed": 0,
        },
        "cross_dataset_analysis": comparison,
        "datasets": [
            para_report,
            r1_report,
        ],
        "failed_files": [],
    }

    # --------------------------------------------------------
    # Generate LLM-ready Markdown
    # --------------------------------------------------------

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt = generate_llm_prompt(final_report)

    output_path.write_text(
        prompt,
        encoding="utf-8",
    )

    # Save structured JSON beside the Markdown report.
    json_path = output_path.with_suffix(".json")

    json_path.write_text(
        json.dumps(
            final_report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("REPORT COMPLETE")
    print("=" * 70)

    print(f"\nLLM-ready Markdown:")
    print(f"  {output_path.absolute()}")

    print(f"\nStructured JSON:")
    print(f"  {json_path.absolute()}")

    print(
        "\nGive dataset_intelligence.md to the LLM."
    )

    print(
        "The report contains both datasets, their comparison, "
        "data-quality findings, leakage considerations, and "
        "instructions for deriving the future ML/production flow."
    )


if __name__ == "__main__":
    main()
