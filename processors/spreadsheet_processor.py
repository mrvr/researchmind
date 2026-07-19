"""
processors/spreadsheet_processor.py
Handles: .csv, .xls, .xlsx, .ods, .tsv

Pipeline:
  1. Load file into pandas DataFrame
  2. Generate statistical description + column analysis
  3. Send description to local LLM → data insights summary
"""

from pathlib import Path
import pandas as pd
import numpy as np
from utils.logger import get_logger
from config import SPREADSHEET_ANALYSIS_PROMPT

log = get_logger("SpreadsheetProcessor")


def _load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load any supported spreadsheet format into a pandas DataFrame."""
    ext = filepath.suffix.lower()

    if ext == ".csv":
        # Try to detect separator automatically
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(str(filepath), sep=sep, encoding_errors="replace")
                if df.shape[1] > 1:  # More than one column = correct separator
                    return df
            except Exception:
                continue
        return pd.read_csv(str(filepath), encoding_errors="replace")

    elif ext == ".tsv":
        return pd.read_csv(str(filepath), sep="\t", encoding_errors="replace")

    elif ext in (".xls",):
        return pd.read_excel(str(filepath), engine="xlrd")

    elif ext in (".xlsx",):
        return pd.read_excel(str(filepath), engine="openpyxl")

    elif ext == ".ods":
        return pd.read_excel(str(filepath), engine="odf")

    else:
        raise ValueError(f"Unsupported spreadsheet format: {ext}")


def _build_data_description(df: pd.DataFrame, filename: str) -> str:
    """
    Build a comprehensive textual description of the DataFrame
    suitable for LLM analysis.
    """
    lines = []

    # Basic info
    lines.append(f"FILE: {filename}")
    lines.append(f"SHAPE: {df.shape[0]:,} rows × {df.shape[1]} columns")
    lines.append(f"COLUMNS: {', '.join(df.columns.tolist())}")
    lines.append("")

    # Column-level analysis
    lines.append("=== COLUMN TYPES ===")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean() * 100
        unique = df[col].nunique()
        lines.append(f"  {col}: {dtype} | {unique} unique values | {null_pct:.1f}% null")

    lines.append("")

    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        lines.append("=== NUMERIC STATISTICS ===")
        desc = df[numeric_cols].describe().round(4)
        lines.append(desc.to_string())
        lines.append("")

        # Correlation hints (if multiple numeric cols)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            # Find top 5 strongest correlations (exclude self-correlations)
            corr_pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(key=abs, ascending=False)
                .head(5)
            )
            if not corr_pairs.empty:
                lines.append("=== TOP CORRELATIONS ===")
                for (c1, c2), val in corr_pairs.items():
                    lines.append(f"  {c1} ↔ {c2}: r = {val:.3f}")
                lines.append("")

    # Categorical summary (top values for string cols)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        lines.append("=== CATEGORICAL COLUMNS (Top 5 values each) ===")
        for col in cat_cols[:5]:  # Limit to first 5 categorical cols
            top = df[col].value_counts().head(5)
            lines.append(f"\n  {col}:")
            for val, cnt in top.items():
                lines.append(f"    '{val}': {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # Sample rows
    lines.append("\n=== SAMPLE DATA (first 5 rows) ===")
    lines.append(df.head(5).to_string())

    return "\n".join(lines)


def generate_insights(description: str, llm_client) -> str:
    """Generate data insights from the description using the local LLM."""
    prompt = SPREADSHEET_ANALYSIS_PROMPT.format(data_description=description[:5000])
    return llm_client.generate(prompt)


def process(filepath: str | Path, llm_client=None) -> dict:
    """
    Full spreadsheet processing pipeline: load → analyze → insights.

    Returns:
        {
            "source": str,
            "type": "spreadsheet",
            "shape": [rows, cols],
            "columns": list[str],
            "description": str,       ← full statistical description
            "insights": str,          ← LLM-generated data insights
            "char_count": int,
            "error": str | None
        }
    """
    filepath = Path(filepath)
    log.info(f"Processing spreadsheet: {filepath.name}")

    result = {
        "source": filepath.name,
        "type": "spreadsheet",
        "shape": [0, 0],
        "columns": [],
        "description": "",
        "insights": "",
        "char_count": 0,
        "error": None,
    }

    try:
        # Load
        df = _load_dataframe(filepath)
        log.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

        result["shape"]   = list(df.shape)
        result["columns"] = df.columns.tolist()

        # Build description
        description = _build_data_description(df, filepath.name)
        result["description"] = description
        result["char_count"]  = len(description)

        # Generate insights via LLM
        if llm_client:
            log.info(f"  Generating data insights via LLM ...")
            result["insights"] = generate_insights(description, llm_client)
            log.info(f"  ✓ Insights generated")
        else:
            result["insights"] = description[:800]

    except Exception as e:
        result["error"] = str(e)
        log.error(f"  ✗ Spreadsheet processing failed: {filepath.name} — {e}")

    return result
