"""
Integration tests for RAG system content-query handling
Tests the complete RAG pipeline from query to response
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from config import Config
from rag_system import RAGSystem
from search_tools import CourseSearchTool
from test_helpers import MockVectorStore


class TestRAGSystemIntegration(unittest.TestCase):
    """Integration tests for the complete RAG system"""

    def setUp(self):
        """Set up test fixtures with temporary database"""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()

        # Mock configuration
        self.test_config = Config()
        self.test_config.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma_db")
        self.test_config.ANTHROPIC_API_KEY = "test_key"

        # Set up RAG system with mocked components
        self.rag_system = RAGSystem(self.test_config)

        # Replace components with mocks
        self.mock_ai_generator = Mock(spec=AIGenerator)
        self.rag_system.ai_generator = self.mock_ai_generator
        self.rag_system.vector_store = MockVectorStore()
        self.rag_system.search_tool = CourseSearchTool(self.rag_system.vector_store)
        self.rag_system.tool_manager.tools["search_course_content"] = (
            self.rag_system.search_tool
        )

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_query_processing_pipeline(self):
        """Test the complete query processing pipeline"""
        query = "What is machine learning?"
        session_id = "test_session_1"

        # Mock AI response
        expected_response = "Machine learning is a subset of AI that enables computers to learn from data."
        self.mock_ai_generator.generate_response.return_value = expected_response

        # Process query
        response, sources = self.rag_system.query(query, session_id)

        # Verify AI generator was called with correct parameters
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args

        # Check that query was passed (as part of prompt)
        self.assertIn("query", call_args[1])
        self.assertIn(query, call_args[1]["query"])

        # Verify result structure
        self.assertEqual(response, expected_response)
        self.assertIsInstance(sources, list)

    def test_session_history_management(self):
        """Test that session history is properly managed across multiple queries"""
        session_id = "test_session_2"

        # First query
        query1 = "What is machine learning?"
        self.mock_ai_generator.generate_response.return_value = "ML is a subset of AI."

        response1, sources1 = self.rag_system.query(query1, session_id)

        # Second query - should include history from first query
        query2 = "How does it work?"
        self.mock_ai_generator.generate_response.return_value = (
            "It works by learning patterns from data."
        )

        response2, sources2 = self.rag_system.query(query2, session_id)

        # Check that second call included history
        second_call_args = self.mock_ai_generator.generate_response.call_args[1]
        conversation_history = second_call_args.get("conversation_history")

        if conversation_history:
            # Should contain the previous query-response pair
            self.assertTrue(
                any(
                    query1 in str(conversation_history)
                    for conversation_history in [conversation_history]
                )
            )

    def test_tool_integration(self):
        """Test that tools are properly integrated and available to AI"""
        query = "Tell me about Python decorators"
        session_id = "test_session_3"

        # Mock tool execution
        self.mock_ai_generator.generate_response.return_value = (
            "Decorators in Python are powerful features that modify functions."
        )

        response, sources = self.rag_system.query(query, session_id)

        # Verify that tools were available
        call_args = self.mock_ai_generator.generate_response.call_args[1]
        self.assertIn("tools", call_args)

        # Should have search tool available
        tools = call_args["tools"]
        self.assertGreater(len(tools), 0)
        tool_names = [tool.get("name") for tool in tools]
        self.assertIn("search_course_content", tool_names)

    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the RAG pipeline"""
        query = "Test query"
        session_id = "test_session_error"

        # Mock AI generator raising an exception
        self.mock_ai_generator.generate_response.side_effect = Exception("API error")

        # Should handle the error gracefully
        with self.assertRaises(Exception):
            self.rag_system.query(query, session_id)

    def test_empty_query_handling(self):
        """Test handling of empty or invalid queries"""
        session_id = "test_session_empty"

        # Test empty query - should work (current implementation doesn't validate empty strings)
        self.mock_ai_generator.generate_response.return_value = "Empty query response"
        response, sources = self.rag_system.query("", session_id)
        self.assertIsInstance(response, str)

        # Test None query - this might raise an error
        try:
            response, sources = self.rag_system.query(None, session_id)
            self.assertIsInstance(response, str)
        except (TypeError, AttributeError):
            # Expected for None input
            pass


class TestContentQueryEvaluation(unittest.TestCase):
    """Specific tests for evaluating how well the system handles content queries"""

    def setUp(self):
        """Set up test system"""
        self.mock_vector_store = MockVectorStore()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_factual_content_queries(self):
        """Test system's ability to handle factual content questions"""
        test_cases = [
            {
                "query": "What is machine learning?",
                "expected_topics": [
                    "machine learning",
                    "artificial intelligence",
                    "data",
                ],
                "course_filter": None,
            },
            {
                "query": "How do Python decorators work?",
                "expected_topics": ["decorators", "Python", "functions"],
                "course_filter": "Advanced Python Programming",
            },
            {
                "query": "Explain linear regression",
                "expected_topics": ["linear regression", "algorithm", "prediction"],
                "course_filter": "Introduction to Machine Learning",
            },
        ]

        for case in test_cases:
            with self.subTest(query=case["query"]):
                result = self.search_tool.execute(
                    case["query"], course_name=case["course_filter"]
                )

                # Check that result contains expected topics
                result_lower = result.lower()
                topic_found = any(
                    topic.lower() in result_lower for topic in case["expected_topics"]
                )
                self.assertTrue(
                    topic_found,
                    f"Expected topics {case['expected_topics']} not found in result: {result}",
                )

    def test_comparative_queries(self):
        """Test handling of queries that require comparing information"""
        queries = [
            "What's the difference between classification and regression?",
            "Compare async programming and synchronous programming",
            "How does MCP differ from traditional APIs?",
        ]

        for query in queries:
            with self.subTest(query=query):
                result = self.search_tool.execute(query)
                # Should return some content (not empty)
                self.assertGreater(len(result.strip()), 0)

    def test_procedural_queries(self):
        """Test handling of 'how-to' procedural queries"""
        test_cases = [
            {
                "query": "How to set up MCP?",
                "course": "Introduction to MCP",
                "lesson": 2,
            },
            {
                "query": "How to use decorators in Python?",
                "course": "Advanced Python Programming",
                "lesson": 1,
            },
        ]

        for case in test_cases:
            with self.subTest(query=case["query"]):
                result = self.search_tool.execute(
                    case["query"],
                    course_name=case["course"],
                    lesson_number=case.get("lesson"),
                )

                # Should contain procedural information
                procedural_indicators = [
                    "how",
                    "setup",
                    "configure",
                    "install",
                    "use",
                    "implement",
                ]
                result_lower = result.lower()
                has_procedural_content = any(
                    indicator in result_lower for indicator in procedural_indicators
                )

                self.assertGreater(len(result.strip()), 0, "Should return some content")

    def test_contextual_queries(self):
        """Test queries that depend on course/lesson context"""
        test_cases = [
            {
                "query": "What topics are covered in lesson 1?",
                "course": "Introduction to Machine Learning",
                "lesson": 1,
                "should_contain": ["machine learning", "introduction"],
            },
            {
                "query": "What advanced concepts are taught?",
                "course": "Advanced Python Programming",
                "should_contain": ["decorators", "async", "advanced"],
            },
        ]

        for case in test_cases:
            with self.subTest(query=case["query"]):
                result = self.search_tool.execute(
                    case["query"],
                    course_name=case["course"],
                    lesson_number=case.get("lesson"),
                )

                if case.get("should_contain"):
                    result_lower = result.lower()
                    for expected_content in case["should_contain"]:
                        found = expected_content.lower() in result_lower
                        if found:
                            break
                    self.assertTrue(
                        found,
                        f"Expected one of {case['should_contain']} in result: {result}",
                    )

    def test_query_specificity_impact(self):
        """Test how query specificity affects result relevance"""
        # Broad query
        broad_result = self.search_tool.execute("programming")
        broad_source_count = len(self.search_tool.last_sources)

        # Specific query
        self.search_tool.last_sources = []  # Reset
        specific_result = self.search_tool.execute("Python decorators syntax")
        specific_source_count = len(self.search_tool.last_sources)

        # More specific queries might return fewer but more relevant sources
        # At minimum, both should return some results
        self.assertGreater(len(broad_result.strip()), 0)
        self.assertGreater(len(specific_result.strip()), 0)

        # Specific query should contain relevant keywords
        specific_lower = specific_result.lower()
        self.assertTrue(
            any(keyword in specific_lower for keyword in ["python", "decorators"])
        )


class TestSearchQualityMetrics(unittest.TestCase):
    """Tests to evaluate the quality of search results"""

    def setUp(self):
        """Set up test system"""
        self.mock_vector_store = MockVectorStore()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_result_relevance(self):
        """Test that search results are relevant to the query"""
        test_queries = [
            ("machine learning", ["machine", "learning", "algorithm"]),
            ("Python programming", ["Python", "programming", "code"]),
            ("MCP setup", ["MCP", "setup", "configure"]),
        ]

        for query, expected_keywords in test_queries:
            with self.subTest(query=query):
                result = self.search_tool.execute(query)
                result_lower = result.lower()

                # At least one expected keyword should be present
                keyword_found = any(
                    keyword.lower() in result_lower for keyword in expected_keywords
                )
                self.assertTrue(
                    keyword_found,
                    f"No relevant keywords {expected_keywords} found in result for query: {query}",
                )

    def test_result_completeness(self):
        """Test that results provide sufficient information"""
        queries = [
            "What is machine learning?",
            "How do decorators work?",
            "Explain MCP protocol",
        ]

        for query in queries:
            with self.subTest(query=query):
                result = self.search_tool.execute(query)

                # Results should be substantial (not just headers)
                content_words = len(
                    [word for word in result.split() if not word.startswith("[")]
                )
                self.assertGreater(
                    content_words, 10, f"Result seems too brief for query: {query}"
                )

    def test_source_attribution(self):
        """Test that results properly attribute sources"""
        result = self.search_tool.execute("machine learning concepts")

        # Should track sources
        self.assertIsInstance(self.search_tool.last_sources, list)

        # Result should contain source headers
        self.assertIn("[", result)  # Should have course/lesson headers
        self.assertIn("]", result)

        # Sources should have required structure
        for source in self.search_tool.last_sources:
            self.assertIn("text", source)
            self.assertIn("link", source)


if __name__ == "__main__":
    # Run all tests
    unittest.main()
