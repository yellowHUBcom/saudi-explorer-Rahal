"""Focused integration tests for the Data and RAG module."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb

from ingest import CHROMA_PATH, COLLECTION_NAME, load_and_validate_data
from rag import retrieve_context


class DataAndRagTests(unittest.TestCase):
    """Verify dataset validation, persistence, filtering, and retrieval quality."""

    @classmethod
    def setUpClass(cls) -> None:
        """Connect once to the already-ingested local collection."""
        cls.collection = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        ).get_collection(COLLECTION_NAME)

    def test_curated_dataset_shape(self) -> None:
        """The curated data has exactly four valid rows per destination."""
        data = load_and_validate_data()
        self.assertEqual(len(data), 16)
        self.assertEqual(set(data.groupby("destination").size()), {4})

    def test_persisted_document_counts(self) -> None:
        """Chroma contains 16 documents and four per destination."""
        self.assertEqual(self.collection.count(), 16)
        for destination in (
            "Riyadh",
            "Jeddah",
            "Abha",
            "Eastern Province",
        ):
            result = self.collection.get(where={"destination": destination})
            self.assertEqual(len(result["ids"]), 4)

    def test_arabic_retrieval_quality(self) -> None:
        """Arabic queries retrieve an expected relevant attraction."""
        cases = [
            (
                "ما أبرز المعالم التاريخية في الرياض؟",
                "Riyadh",
                {"حي الطريف التاريخي", "قصر المصمك"},
            ),
            (
                "ما الأماكن المناسبة للتنزه قرب البحر في جدة؟",
                "Jeddah",
                {"واجهة جدة البحرية"},
            ),
            (
                "أريد مكانًا طبيعيًا في أبها",
                "Abha",
                {"جبل السودة", "منتزه عسير الوطني"},
            ),
            (
                "ما الأماكن الثقافية في المنطقة الشرقية؟",
                "Eastern Province",
                {"مركز الملك عبدالعزيز الثقافي العالمي — إثراء"},
            ),
        ]
        for query, destination, expected_places in cases:
            with self.subTest(destination=destination):
                results = retrieve_context(query, destination)
                returned_places = {item["place_name"] for item in results}
                self.assertTrue(returned_places & expected_places)
                self.assertTrue(
                    all(item["destination"] == destination for item in results)
                )
                self.assertTrue(
                    all(
                        item["source_name"]
                        and item["source_section"]
                        and item["source_url"]
                        for item in results
                    )
                )

    def test_invalid_inputs_return_empty_lists(self) -> None:
        """Normal invalid input is handled without raising exceptions."""
        self.assertEqual(retrieve_context("سؤال", "Makkah"), [])
        self.assertEqual(retrieve_context(" ", "Riyadh"), [])
        self.assertEqual(retrieve_context("سؤال", "Riyadh", 0), [])

    def test_missing_database_returns_empty_list(self) -> None:
        """An unavailable Chroma directory does not crash the caller."""
        missing_path = Path("tests") / "does-not-exist-chroma"
        with patch("rag.CHROMA_PATH", missing_path):
            self.assertEqual(retrieve_context("أماكن تاريخية", "Riyadh"), [])

    def test_unrelated_queries_return_empty_lists(self) -> None:
        """Low-evidence queries are removed by the distance threshold."""
        cases = [
            ("ما أفضل المستشفيات لعلاج القلب؟", "Riyadh"),
            ("ما أسعار تذاكر الطيران؟", "Jeddah"),
            ("كيف أقدم على وظيفة؟", "Abha"),
        ]
        for query, destination in cases:
            with self.subTest(query=query):
                self.assertEqual(retrieve_context(query, destination), [])


if __name__ == "__main__":
    unittest.main()
