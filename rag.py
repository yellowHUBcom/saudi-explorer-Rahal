"""Retrieve destination-filtered context from the local Chroma database."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "travel_destinations"
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
SUPPORTED_DESTINATIONS = {
    "Riyadh",
    "Jeddah",
    "Abha",
    "Eastern Province",
}
# Calibration: expected Arabic attraction matches were at or below 0.7623,
# while the closest of three clearly unrelated queries was 0.8255.
MAX_RETRIEVAL_DISTANCE = 0.78

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_embedding_model() -> SentenceTransformer:
    """Load the multilingual embedding model once per process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def _is_valid_request(
    query: object,
    destination: object,
    top_k: object,
) -> bool:
    """Return whether retrieval inputs satisfy the public contract."""
    return (
        isinstance(query, str)
        and bool(query.strip())
        and isinstance(destination, str)
        and destination in SUPPORTED_DESTINATIONS
        and isinstance(top_k, int)
        and not isinstance(top_k, bool)
        and top_k > 0
    )


def _open_collection():
    """Open the existing persistent collection without creating data."""
    if not CHROMA_PATH.is_dir() or not any(CHROMA_PATH.iterdir()):
        return None

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection_names = {
        collection
        if isinstance(collection, str)
        else collection.name
        for collection in client.list_collections()
    }
    if COLLECTION_NAME not in collection_names:
        return None
    return client.get_collection(COLLECTION_NAME)


def retrieve_context(
    query: str,
    destination: str,
    top_k: int = 4
) -> list[dict]:
    """Return relevant stored context for an exact supported destination.

    Invalid input, unavailable storage, missing collections, and empty or failed
    retrievals return an empty list so callers can handle insufficient evidence.
    """
    if not _is_valid_request(query, destination, top_k):
        return []

    try:
        collection = _open_collection()
        if collection is None:
            return []

        query_embedding = _load_embedding_model().encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 4),
            where={"destination": destination},
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        logger.exception("RAG retrieval failed")
        return []

    documents = response.get("documents") or [[]]
    metadatas = response.get("metadatas") or [[]]
    distances = response.get("distances") or [[]]
    if not documents[0]:
        return []

    results: list[dict] = []
    for text, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        numeric_distance = float(distance)
        if (
            metadata.get("destination") != destination
            or numeric_distance > MAX_RETRIEVAL_DISTANCE
        ):
            continue
        results.append(
            {
                "text": text,
                "destination": metadata.get("destination", ""),
                "place_name": metadata.get("place_name", ""),
                "category": metadata.get("category", ""),
                "source_name": metadata.get("source_name", ""),
                "source_section": metadata.get("source_section", ""),
                "source_url": metadata.get("source_url", ""),
                "distance": numeric_distance,
            }
        )
    return results
