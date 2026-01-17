from pathlib import Path
import pandas as pd

def load_csv(csv_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding=encoding)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df

def validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available: {list(df.columns)}")
