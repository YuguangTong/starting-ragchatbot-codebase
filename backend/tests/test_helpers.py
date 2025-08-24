"""Test helpers and mocks for RAG system tests"""

import os
import sys
from typing import List, Optional

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixtures import SAMPLE_CHUNKS
from vector_store import SearchResults


class MockVectorStore:
    """Mock VectorStore for testing CourseSearchTool without actual database"""

    def __init__(self, scenario: str = "default"):
        self.scenario = scenario
        self.search_calls = []
        self.lesson_links = {
            ("Introduction to Machine Learning", 1): "https://example.com/ml-lesson1",
            ("Introduction to Machine Learning", 2): "https://example.com/ml-lesson2",
            ("Introduction to Machine Learning", 3): "https://example.com/ml-lesson3",
            ("Introduction to MCP", 1): "https://example.com/mcp-lesson1",
            ("Introduction to MCP", 2): "https://example.com/mcp-lesson2",
            ("Advanced Python Programming", 1): "https://example.com/python-lesson1",
            ("Advanced Python Programming", 2): "https://example.com/python-lesson2",
            ("Advanced Python Programming", 3): "https://example.com/python-lesson3",
        }

    def search(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> SearchResults:
        """Mock search method that returns predefined results based on scenario"""

        # Record the search call for verification
        self.search_calls.append(
            {
                "query": query,
                "course_name": course_name,
                "lesson_number": lesson_number,
                "limit": limit,
            }
        )

        # Handle different test scenarios
        if self.scenario == "error":
            return SearchResults.empty("Mock database error")

        if self.scenario == "empty":
            return SearchResults(documents=[], metadata=[], distances=[])

        if self.scenario == "course_not_found":
            if course_name and "Nonexistent" in course_name:
                return SearchResults.empty(f"No course found matching '{course_name}'")

        # Default behavior - return relevant chunks based on query and filters
        relevant_chunks = []

        for chunk in SAMPLE_CHUNKS:
            # Check if chunk matches the filters
            if course_name:
                # Fuzzy course name matching
                course_matches = (
                    course_name.lower() in chunk.course_title.lower()
                    or chunk.course_title.lower() in course_name.lower()
                    or any(
                        word in chunk.course_title.lower()
                        for word in course_name.lower().split()
                    )
                )
                if not course_matches:
                    continue

            if lesson_number is not None:
                if chunk.lesson_number != lesson_number:
                    continue

            # Check if chunk content is relevant to the query
            query_words = query.lower().split()
            content_lower = chunk.content.lower()

            if any(word in content_lower for word in query_words):
                relevant_chunks.append(chunk)

        # Sort by relevance (simple scoring based on query word matches)
        def relevance_score(chunk):
            query_words = query.lower().split()
            content_lower = chunk.content.lower()
            return sum(1 for word in query_words if word in content_lower)

        relevant_chunks.sort(key=relevance_score, reverse=True)

        # Limit results
        max_results = limit if limit is not None else 5
        relevant_chunks = relevant_chunks[:max_results]

        if not relevant_chunks:
            return SearchResults(documents=[], metadata=[], distances=[])

        # Create SearchResults
        documents = [chunk.content for chunk in relevant_chunks]
        metadata = []
        distances = [0.1 * i for i in range(len(documents))]

        for chunk in relevant_chunks:
            meta = {"course_title": chunk.course_title}
            if chunk.lesson_number is not None:
                meta["lesson_number"] = chunk.lesson_number
            metadata.append(meta)

        return SearchResults(
            documents=documents, metadata=metadata, distances=distances
        )

    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        """Mock method to return lesson links"""
        return self.lesson_links.get((course_title, lesson_number))

    def reset_calls(self):
        """Reset the search calls list for testing"""
        self.search_calls = []


class MockSearchResultsBuilder:
    """Helper class to build mock SearchResults for specific test cases"""

    @staticmethod
    def create_ml_results() -> SearchResults:
        """Create results for machine learning query"""
        return SearchResults(
            documents=[
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                "Linear regression is one of the simplest machine learning algorithms for predictive modeling.",
            ],
            metadata=[
                {
                    "course_title": "Introduction to Machine Learning",
                    "lesson_number": 1,
                },
                {
                    "course_title": "Introduction to Machine Learning",
                    "lesson_number": 2,
                },
            ],
            distances=[0.1, 0.2],
        )

    @staticmethod
    def create_mcp_results() -> SearchResults:
        """Create results for MCP query"""
        return SearchResults(
            documents=[
                "MCP (Model Context Protocol) enables AI models to connect to external data sources."
            ],
            metadata=[{"course_title": "Introduction to MCP", "lesson_number": 1}],
            distances=[0.05],
        )

    @staticmethod
    def create_empty_results() -> SearchResults:
        """Create empty results"""
        return SearchResults(documents=[], metadata=[], distances=[])

    @staticmethod
    def create_error_results(error_msg: str) -> SearchResults:
        """Create error results"""
        return SearchResults.empty(error_msg)


def assert_search_result_format(
    result: str, expected_courses: List[str], expected_lessons: List[int] = None
):
    """Helper function to assert the format of search results"""
    if expected_lessons is None:
        expected_lessons = [None] * len(expected_courses)

    # Check that the result contains course information
    for i, course in enumerate(expected_courses):
        assert f"[{course}" in result, f"Expected course '{course}' not found in result"

        if expected_lessons[i] is not None:
            assert f"Lesson {expected_lessons[i]}" in result, (
                f"Expected lesson {expected_lessons[i]} not found in result"
            )


def count_sources_in_result(result: str) -> int:
    """Count the number of source headers in a formatted result"""
    import re

    # Count occurrences of pattern [Course Title - Lesson X] or [Course Title]
    pattern = r"\[[^\]]+\]"
    matches = re.findall(pattern, result)
    return len(matches)


def extract_content_from_result(result: str) -> List[str]:
    """Extract the actual content (not headers) from a formatted result"""
    lines = result.split("\n")
    content_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines and header lines that start with [
        if line and not line.startswith("["):
            content_lines.append(line)

    return content_lines
