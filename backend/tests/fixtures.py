"""Test fixtures and mock data for RAG system tests"""

import os
import sys
from typing import List

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults

# Sample courses for testing
SAMPLE_COURSES = [
    Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Sarah Johnson",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is Machine Learning?",
                lesson_link="https://example.com/ml-lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Linear Regression Basics",
                lesson_link="https://example.com/ml-lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Classification Algorithms",
                lesson_link="https://example.com/ml-lesson3",
            ),
        ],
    ),
    Course(
        title="Introduction to MCP",
        course_link="https://example.com/mcp-course",
        instructor="Claude AI",
        lessons=[
            Lesson(
                lesson_number=1,
                title="MCP Overview",
                lesson_link="https://example.com/mcp-lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Setting up MCP",
                lesson_link="https://example.com/mcp-lesson2",
            ),
        ],
    ),
    Course(
        title="Advanced Python Programming",
        course_link="https://example.com/python-course",
        instructor="Prof. Alex Chen",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Decorators and Context Managers",
                lesson_link="https://example.com/python-lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Async Programming",
                lesson_link="https://example.com/python-lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Testing Best Practices",
                lesson_link="https://example.com/python-lesson3",
            ),
        ],
    ),
]

# Sample course chunks for testing
SAMPLE_CHUNKS = [
    CourseChunk(
        content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications based on those patterns.",
        course_title="Introduction to Machine Learning",
        lesson_number=1,
        chunk_index=0,
    ),
    CourseChunk(
        content="Linear regression is one of the simplest machine learning algorithms. It models the relationship between a dependent variable and independent variables by fitting a linear equation to observed data. The goal is to find the best line that minimizes the sum of squared residuals.",
        course_title="Introduction to Machine Learning",
        lesson_number=2,
        chunk_index=1,
    ),
    CourseChunk(
        content="Classification algorithms are used to predict discrete categories or classes. Popular algorithms include logistic regression, decision trees, random forests, and support vector machines. Each has different strengths depending on the data and problem type.",
        course_title="Introduction to Machine Learning",
        lesson_number=3,
        chunk_index=2,
    ),
    CourseChunk(
        content="MCP (Model Context Protocol) is a protocol that enables AI models to securely connect to external data sources and tools. It provides a standardized way for language models to access real-time information and perform actions beyond their training data.",
        course_title="Introduction to MCP",
        lesson_number=1,
        chunk_index=3,
    ),
    CourseChunk(
        content="To set up MCP, you need to configure the server-client architecture. The MCP server hosts the tools and data sources, while clients (like Claude) can connect to access these resources. This involves proper authentication and protocol implementation.",
        course_title="Introduction to MCP",
        lesson_number=2,
        chunk_index=4,
    ),
    CourseChunk(
        content="Decorators in Python are a powerful feature that allows you to modify or enhance functions without changing their code. They follow the @decorator syntax and are commonly used for logging, authentication, caching, and other cross-cutting concerns.",
        course_title="Advanced Python Programming",
        lesson_number=1,
        chunk_index=5,
    ),
    CourseChunk(
        content="Async programming in Python uses the asyncio library to handle concurrent operations efficiently. It allows you to write non-blocking code using async/await syntax, making it perfect for I/O-bound operations like web requests or database queries.",
        course_title="Advanced Python Programming",
        lesson_number=2,
        chunk_index=6,
    ),
]


# Mock SearchResults for different scenarios
def create_search_results(
    documents: List[str],
    course_titles: List[str],
    lesson_numbers: List[int] = None,
    distances: List[float] = None,
) -> SearchResults:
    """Create mock SearchResults for testing"""
    if lesson_numbers is None:
        lesson_numbers = [None] * len(documents)
    if distances is None:
        distances = [0.1 * i for i in range(len(documents))]

    metadata = []
    for i, (course_title, lesson_num) in enumerate(zip(course_titles, lesson_numbers)):
        meta = {"course_title": course_title}
        if lesson_num is not None:
            meta["lesson_number"] = lesson_num
        metadata.append(meta)

    return SearchResults(documents=documents, metadata=metadata, distances=distances)


# Predefined test scenarios
SEARCH_SCENARIOS = {
    "ml_basic_query": {
        "query": "What is machine learning?",
        "expected_results": create_search_results(
            documents=[SAMPLE_CHUNKS[0].content],
            course_titles=["Introduction to Machine Learning"],
            lesson_numbers=[1],
        ),
    },
    "mcp_query": {
        "query": "Tell me about MCP",
        "course_name": "MCP",
        "expected_results": create_search_results(
            documents=[SAMPLE_CHUNKS[3].content, SAMPLE_CHUNKS[4].content],
            course_titles=["Introduction to MCP", "Introduction to MCP"],
            lesson_numbers=[1, 2],
        ),
    },
    "python_decorators": {
        "query": "How do decorators work?",
        "course_name": "Advanced Python Programming",
        "lesson_number": 1,
        "expected_results": create_search_results(
            documents=[SAMPLE_CHUNKS[5].content],
            course_titles=["Advanced Python Programming"],
            lesson_numbers=[1],
        ),
    },
    "empty_results": {
        "query": "quantum computing",
        "expected_results": SearchResults(documents=[], metadata=[], distances=[]),
    },
    "error_scenario": {
        "query": "test query",
        "expected_results": SearchResults.empty("Database connection error"),
    },
}

# Test case configurations
TEST_CASES = [
    {
        "name": "basic_query_no_filters",
        "query": "machine learning algorithms",
        "course_name": None,
        "lesson_number": None,
        "expected_chunks": 2,
        "should_contain": ["machine learning", "algorithms"],
    },
    {
        "name": "query_with_course_filter",
        "query": "programming concepts",
        "course_name": "Advanced Python Programming",
        "lesson_number": None,
        "expected_chunks": 2,
        "should_contain": ["Python", "programming"],
    },
    {
        "name": "query_with_lesson_filter",
        "query": "regression",
        "course_name": "Introduction to Machine Learning",
        "lesson_number": 2,
        "expected_chunks": 1,
        "should_contain": ["Linear regression", "algorithm"],
    },
    {
        "name": "query_both_filters",
        "query": "decorators",
        "course_name": "Advanced Python Programming",
        "lesson_number": 1,
        "expected_chunks": 1,
        "should_contain": ["Decorators", "Python"],
    },
    {
        "name": "no_results_query",
        "query": "quantum physics",
        "course_name": None,
        "lesson_number": None,
        "expected_chunks": 0,
        "should_contain": [],
    },
    {
        "name": "nonexistent_course",
        "query": "any content",
        "course_name": "Nonexistent Course",
        "lesson_number": None,
        "expected_chunks": 0,
        "should_contain": [],
    },
]
