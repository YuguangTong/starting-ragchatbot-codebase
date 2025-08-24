"""
Comprehensive unit tests for CourseSearchTool.execute method
Tests the core functionality of the RAG system's search capabilities
"""
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, Tool
from vector_store import SearchResults
from test_helpers import MockVectorStore, MockSearchResultsBuilder, assert_search_result_format, count_sources_in_result
from fixtures import TEST_CASES, SAMPLE_CHUNKS

class TestCourseSearchToolExecute(unittest.TestCase):
    """Test the execute method of CourseSearchTool under various conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = MockVectorStore()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_basic_query_no_filters(self):
        """Test basic search without course name or lesson number filters"""
        query = "machine learning algorithms"
        result = self.search_tool.execute(query)
        
        # Check that vector store was called with correct parameters
        self.assertEqual(len(self.mock_vector_store.search_calls), 1)
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call["query"], query)
        self.assertIsNone(call["course_name"])
        self.assertIsNone(call["lesson_number"])
        
        # Check result format and content
        self.assertIsInstance(result, str)
        self.assertIn("machine learning", result.lower())
        
        # Should contain source headers
        source_count = count_sources_in_result(result)
        self.assertGreater(source_count, 0, "Result should contain source information")
    
    def test_query_with_course_filter(self):
        """Test search with course name filter"""
        query = "programming concepts"
        course_name = "Advanced Python Programming"
        
        result = self.search_tool.execute(query, course_name=course_name)
        
        # Check vector store call
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call["course_name"], course_name)
        
        # Result should contain Python-related content
        self.assertIn("Python", result)
        
        # Check source tracking
        self.assertTrue(hasattr(self.search_tool, 'last_sources'))
        self.assertGreater(len(self.search_tool.last_sources), 0)
    
    def test_query_with_lesson_filter(self):
        """Test search with lesson number filter"""
        query = "regression algorithms"
        course_name = "Introduction to Machine Learning" 
        lesson_number = 2
        
        result = self.search_tool.execute(query, course_name=course_name, lesson_number=lesson_number)
        
        # Check vector store call
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call["lesson_number"], lesson_number)
        
        # Result should contain lesson-specific information
        self.assertIn("Lesson 2", result)
        self.assertIn("regression", result.lower())
    
    def test_query_with_both_filters(self):
        """Test search with both course name and lesson number filters"""
        query = "decorators"
        course_name = "Advanced Python Programming"
        lesson_number = 1
        
        result = self.search_tool.execute(query, course_name=course_name, lesson_number=lesson_number)
        
        # Check vector store call
        call = self.mock_vector_store.search_calls[0]
        self.assertEqual(call["course_name"], course_name)
        self.assertEqual(call["lesson_number"], lesson_number)
        
        # Result should be specific to the filtered content
        self.assertIn("Decorators", result)
        self.assertIn("Python", result)
        self.assertIn("Lesson 1", result)
    
    def test_empty_results_no_filters(self):
        """Test handling of empty search results without filters"""
        mock_store = MockVectorStore(scenario="empty")
        tool = CourseSearchTool(mock_store)
        
        result = tool.execute("nonexistent topic")
        
        self.assertEqual(result, "No relevant content found.")
    
    def test_empty_results_with_course_filter(self):
        """Test handling of empty results with course filter"""
        mock_store = MockVectorStore(scenario="empty")
        tool = CourseSearchTool(mock_store)
        
        result = tool.execute("any query", course_name="Some Course")
        
        self.assertEqual(result, "No relevant content found in course 'Some Course'.")
    
    def test_empty_results_with_lesson_filter(self):
        """Test handling of empty results with lesson filter"""
        mock_store = MockVectorStore(scenario="empty")
        tool = CourseSearchTool(mock_store)
        
        result = tool.execute("any query", lesson_number=5)
        
        self.assertEqual(result, "No relevant content found in lesson 5.")
    
    def test_empty_results_with_both_filters(self):
        """Test handling of empty results with both filters"""
        mock_store = MockVectorStore(scenario="empty")
        tool = CourseSearchTool(mock_store)
        
        result = tool.execute("any query", course_name="Some Course", lesson_number=3)
        
        expected = "No relevant content found in course 'Some Course' in lesson 3."
        self.assertEqual(result, expected)
    
    def test_error_handling(self):
        """Test handling of vector store errors"""
        mock_store = MockVectorStore(scenario="error")
        tool = CourseSearchTool(mock_store)
        
        result = tool.execute("any query")
        
        self.assertEqual(result, "Mock database error")
    
    def test_course_not_found_error(self):
        """Test handling when course name doesn't match any existing course"""
        mock_store = MockVectorStore(scenario="course_not_found")
        tool = CourseSearchTool(mock_store)
        
        result = tool.execute("any query", course_name="Nonexistent Course")
        
        self.assertEqual(result, "No course found matching 'Nonexistent Course'")
    
    def test_result_formatting_structure(self):
        """Test the structure and formatting of search results"""
        query = "machine learning"
        result = self.search_tool.execute(query)
        
        # Result should contain properly formatted headers
        lines = result.split('\n')
        header_lines = [line for line in lines if line.strip().startswith('[')]
        content_lines = [line for line in lines if line.strip() and not line.strip().startswith('[')]
        
        self.assertGreater(len(header_lines), 0, "Should have header lines")
        self.assertGreater(len(content_lines), 0, "Should have content lines")
        
        # Each header should be properly formatted
        for header in header_lines:
            self.assertTrue(header.strip().startswith('[') and header.strip().endswith(']'))
    
    def test_source_tracking(self):
        """Test that sources are properly tracked for UI display"""
        query = "python programming"
        course_name = "Advanced Python Programming"
        
        # Clear any previous sources
        self.search_tool.last_sources = []
        
        result = self.search_tool.execute(query, course_name=course_name)
        
        # Check that sources were tracked
        self.assertIsInstance(self.search_tool.last_sources, list)
        
        if len(self.search_tool.last_sources) > 0:
            # Each source should have required fields
            for source in self.search_tool.last_sources:
                self.assertIn("text", source)
                self.assertIn("link", source)
                self.assertIsInstance(source["text"], str)
                # link can be None or a string
                if source["link"] is not None:
                    self.assertIsInstance(source["link"], str)
    
    def test_lesson_links_in_sources(self):
        """Test that lesson links are properly included in sources"""
        query = "decorators"
        course_name = "Advanced Python Programming"
        lesson_number = 1
        
        result = self.search_tool.execute(query, course_name=course_name, lesson_number=lesson_number)
        
        # Check if sources have lesson information
        for source in self.search_tool.last_sources:
            if "Lesson" in source["text"]:
                # Should have attempted to get lesson link
                # In our mock, this should return a link
                self.assertIsNotNone(source["link"])
                self.assertTrue(source["link"].startswith("https://"))
    
    def test_fuzzy_course_name_matching(self):
        """Test that partial course names work (fuzzy matching)"""
        # Test with partial course name
        result = self.search_tool.execute("setup process", course_name="MCP")
        
        # Should find content from "Introduction to MCP" course
        self.assertIn("MCP", result)
        
        # Test with another partial match
        result2 = self.search_tool.execute("algorithms", course_name="Machine Learning")
        self.assertIn("machine learning", result2.lower())
    
    def test_case_insensitive_search(self):
        """Test that search is case insensitive"""
        result1 = self.search_tool.execute("MACHINE LEARNING")
        result2 = self.search_tool.execute("machine learning")
        result3 = self.search_tool.execute("Machine Learning")
        
        # All should return similar results (content should be found)
        # The exact results might differ due to mock behavior, but none should be empty
        self.assertGreater(len(result1), 0)
        self.assertGreater(len(result2), 0)
        self.assertGreater(len(result3), 0)
    
    def test_multiple_results_ordering(self):
        """Test that multiple results are properly formatted and separated"""
        query = "programming"  # Should match multiple chunks
        result = self.search_tool.execute(query)
        
        # Multiple results should be separated by double newlines
        if "\\n\\n" in repr(result):  # Check if there are multiple sections
            sections = result.split('\n\n')
            self.assertGreater(len(sections), 1, "Multiple results should be separated")
            
            # Each section should have a header
            for section in sections:
                if section.strip():
                    lines = section.strip().split('\n')
                    self.assertTrue(lines[0].strip().startswith('['), 
                                   f"Each section should start with a header: {lines[0]}")

class TestCourseSearchToolInterface(unittest.TestCase):
    """Test that CourseSearchTool properly implements the Tool interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = MockVectorStore()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_implements_tool_interface(self):
        """Test that CourseSearchTool implements the Tool interface"""
        self.assertIsInstance(self.search_tool, Tool)
        
        # Should have required methods
        self.assertTrue(hasattr(self.search_tool, 'get_tool_definition'))
        self.assertTrue(hasattr(self.search_tool, 'execute'))
        self.assertTrue(callable(self.search_tool.get_tool_definition))
        self.assertTrue(callable(self.search_tool.execute))
    
    def test_tool_definition_structure(self):
        """Test the structure of the tool definition"""
        definition = self.search_tool.get_tool_definition()
        
        self.assertIsInstance(definition, dict)
        self.assertIn("name", definition)
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        
        # Check input schema structure
        schema = definition["input_schema"]
        self.assertIn("type", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        # Check required properties
        properties = schema["properties"]
        self.assertIn("query", properties)
        self.assertIn("course_name", properties) 
        self.assertIn("lesson_number", properties)
        
        # Query should be required
        self.assertIn("query", schema["required"])
    
    def test_initialization_with_vector_store(self):
        """Test proper initialization with vector store"""
        mock_store = MockVectorStore()
        tool = CourseSearchTool(mock_store)
        
        self.assertEqual(tool.store, mock_store)
        self.assertEqual(tool.last_sources, [])

if __name__ == '__main__':
    unittest.main()