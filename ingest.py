"""Build the persistent Chroma knowledge base from the curated CSV file."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "destinations.csv"
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "travel_destinations"
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

EXPECTED_COLUMNS = [
    "destination",
    "place_name",
    "category",
    "description",
    "recommended_duration",
    "suitable_for",
    "activity_level",
    "source_name",
    "source_section",
    "source_url",
]
SUPPORTED_DESTINATIONS = {
    "Riyadh",
    "Jeddah",
    "Abha",
    "Eastern Province",
}
ALLOWED_CATEGORIES = {
    "معلومات عامة",
    "معلم بارز",
    "تاريخي",
    "ثقافي",
    "طبيعي",
    "واجهة بحرية",
}
ALLOWED_DURATIONS = {
    "ساعة واحدة",
    "ساعتان",
    "3 ساعات",
    "نصف يوم",
    "يوم كامل",
    "عدة أيام",
}
ALLOWED_AUDIENCES = {
    "العائلات",
    "الأصدقاء",
    "الأفراد",
    "العائلات والأصدقاء",
    "العائلات والأفراد",
    "الأصدقاء والأفراد",
    "جميع المسافرين",
}
ALLOWED_ACTIVITY_LEVELS = {"منخفض", "متوسط", "مرتفع"}


def _normalize_whitespace(value: object) -> str:
    """Trim a value and collapse repeated whitespace without changing wording."""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_and_validate_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load, clean in memory, and validate the curated destination dataset."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    data = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    if list(data.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "CSV columns must exactly match this order: "
            + ", ".join(EXPECTED_COLUMNS)
        )
    if len(data) != 16:
        raise ValueError(f"Dataset must contain exactly 16 rows; found {len(data)}")
    if data.isna().any().any():
        raise ValueError("Dataset contains null values")

    data = data.apply(lambda column: column.map(_normalize_whitespace))
    if data.eq("").any().any():
        raise ValueError("Dataset contains empty values")
    if data.duplicated().any():
        raise ValueError("Dataset contains duplicate rows")

    destinations = set(data["destination"])
    if destinations != SUPPORTED_DESTINATIONS:
        raise ValueError(
            "Destination identifiers must be exactly: "
            + ", ".join(sorted(SUPPORTED_DESTINATIONS))
        )

    _validate_allowed_values(data, "category", ALLOWED_CATEGORIES)
    _validate_allowed_values(
        data, "recommended_duration", ALLOWED_DURATIONS
    )
    _validate_allowed_values(data, "suitable_for", ALLOWED_AUDIENCES)
    _validate_allowed_values(
        data, "activity_level", ALLOWED_ACTIVITY_LEVELS
    )

    for destination, group in data.groupby("destination"):
        overview_count = int((group["category"] == "معلومات عامة").sum())
        if len(group) != 4 or overview_count != 1:
            raise ValueError(
                f"{destination} must have one overview and three attractions"
            )

    return data


def _validate_allowed_values(
    data: pd.DataFrame, column: str, allowed: set[str]
) -> None:
    """Ensure a controlled column contains only approved values."""
    invalid = sorted(set(data[column]) - allowed)
    if invalid:
        raise ValueError(f"Invalid {column} values: {invalid}")


def _create_document(row: pd.Series) -> str:
    """Create one coherent Arabic retrieval document from a dataset row."""
    return "\n".join(
        [
            f"الوجهة: {row['destination']}",
            f"اسم المكان: {row['place_name']}",
            f"التصنيف: {row['category']}",
            f"الوصف: {row['description']}",
            f"المدة المقترحة: {row['recommended_duration']}",
            f"مناسب لـ: {row['suitable_for']}",
            f"مستوى النشاط: {row['activity_level']}",
        ]
    )


def _create_record_id(row: pd.Series) -> str:
    """Return a stable identifier for a destination/place pair."""
    key = f"{row['destination']}|{row['place_name']}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def _create_metadata(row: pd.Series) -> dict[str, str]:
    """Create Chroma metadata for one dataset row."""
    fields = [column for column in EXPECTED_COLUMNS if column != "description"]
    metadata = {field: row[field] for field in fields}
    metadata["record_type"] = (
        "overview" if row["category"] == "معلومات عامة" else "attraction"
    )
    return metadata


def ingest() -> dict[str, int]:
    """Validate the dataset and atomically rebuild its Chroma collection."""
    data = load_and_validate_data()
    documents = [_create_document(row) for _, row in data.iterrows()]
    metadatas = [_create_metadata(row) for _, row in data.iterrows()]
    ids = [_create_record_id(row) for _, row in data.iterrows()]

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection_names = {item.name for item in client.list_collections()}
    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    if collection.count() != 16:
        raise RuntimeError(
            f"Expected 16 stored documents; found {collection.count()}"
        )
    return {key: int(value) for key, value in data["destination"].value_counts().items()}


def main() -> int:
    """Run ingestion and print a concise success or failure summary."""
    try:
        destination_counts = ingest()
    except Exception as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    print("Ingestion completed successfully")
    print("Rows: 16")
    print(f"Destination counts: {destination_counts}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
