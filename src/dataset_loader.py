"""Dataset loader and validator for T20 Cricket World Cup News.

Handles dataset ingestion, column normalization, deduplication, missing value
handling, and schema validation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "t20_cricket_news.csv"
FALLBACK_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_articles.csv"

REQUIRED_COLUMNS = ["article_id", "title", "article_text", "category", "source", "published_date"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and map common variations to expected schema."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map text variants
    text_mapping = {
        "text": "article_text",
        "content": "article_text",
        "body": "article_text",
        "article": "article_text",
        "news": "article_text",
        "description": "article_text",
    }
    for old_col, new_col in text_mapping.items():
        if old_col in df.columns and "article_text" not in df.columns:
            df = df.rename(columns={old_col: new_col})
            break

    # Map date variants
    date_mapping = {"date": "published_date", "pub_date": "published_date", "time": "published_date"}
    for old_col, new_col in date_mapping.items():
        if old_col in df.columns and "published_date" not in df.columns:
            df = df.rename(columns={old_col: new_col})
            break

    # Map id variants
    id_mapping = {"id": "article_id", "news_id": "article_id"}
    for old_col, new_col in id_mapping.items():
        if old_col in df.columns and "article_id" not in df.columns:
            df = df.rename(columns={old_col: new_col})
            break

    # Map category variants
    cat_mapping = {"topic": "category", "label": "category", "section": "category"}
    for old_col, new_col in cat_mapping.items():
        if old_col in df.columns and "category" not in df.columns:
            df = df.rename(columns={old_col: new_col})
            break

    return df


def validate_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns, drop invalid/empty rows, and remove duplicates."""
    df = normalize_columns(df)

    if "article_text" not in df.columns:
        raise ValueError("Dataset must contain an 'article_text' (or 'text'/'content') column.")

    if "title" not in df.columns:
        df["title"] = [f"Article {i + 1}" for i in range(len(df))]

    if "article_id" not in df.columns:
        df["article_id"] = [f"T20-{i + 1:03d}" for i in range(len(df))]

    if "category" not in df.columns:
        df["category"] = "General Cricket"

    if "source" not in df.columns:
        df["source"] = "Curated Source"

    if "published_date" not in df.columns:
        df["published_date"] = "2024-06-29"

    # Fill nulls and clean strings
    df["title"] = df["title"].fillna("Untitled Article").astype(str).str.strip()
    df["article_text"] = df["article_text"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].fillna("General Cricket").astype(str).str.strip()
    df["source"] = df["source"].fillna("Curated Source").astype(str).str.strip()
    df["published_date"] = df["published_date"].fillna("2024-06-29").astype(str).str.strip()
    df["article_id"] = df["article_id"].fillna("").astype(str).str.strip()

    # Drop empty articles
    df = df[df["article_text"].str.len() > 15].copy()

    # Remove duplicates based on article_text
    df = df.drop_duplicates(subset=["article_text"]).reset_index(drop=True)

    # Re-order and select required columns
    ordered_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[ordered_cols + remaining_cols]

    return df


def load_dataset(file_or_path: Optional[Union[str, Path, object]] = None) -> pd.DataFrame:
    """Load and validate dataset from file path or file-like object."""
    if file_or_path is not None:
        if isinstance(file_or_path, (str, Path)):
            path = Path(file_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found at {path}")
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(file_or_path)
    else:
        if DEFAULT_DATA_PATH.exists():
            df = pd.read_csv(DEFAULT_DATA_PATH)
        elif FALLBACK_DATA_PATH.exists():
            df = pd.read_csv(FALLBACK_DATA_PATH)
        else:
            raise FileNotFoundError(
                f"No default dataset found at {DEFAULT_DATA_PATH} or {FALLBACK_DATA_PATH}."
            )

    return validate_and_clean_dataset(df)


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Union[int, float, list]]:
    """Compute summary statistics for the dataset."""
    word_counts = df["article_text"].apply(lambda t: len(str(t).split()))
    stats = {
        "total_articles": int(len(df)),
        "unique_categories": int(df["category"].nunique()) if "category" in df.columns else 0,
        "categories_list": sorted(df["category"].unique().tolist()) if "category" in df.columns else [],
        "unique_sources": int(df["source"].nunique()) if "source" in df.columns else 0,
        "min_word_count": int(word_counts.min()) if not word_counts.empty else 0,
        "max_word_count": int(word_counts.max()) if not word_counts.empty else 0,
        "avg_word_count": float(round(word_counts.mean(), 1)) if not word_counts.empty else 0.0,
        "median_word_count": float(round(word_counts.median(), 1)) if not word_counts.empty else 0.0,
    }
    return stats
