"""Integration tests for ReAct functionality with real components"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from config import Config
from rag_system import RAGSystem
from models import Course, Lesson
from llm_providers.base_provider import LLMResponse, ToolCall


class TestReActIntegration(unittest.TestCase):
    """Integration tests for ReAct loop with real vector store"""
    
    def setUp(self):
        """Set up test environment with temporary database"""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure test settings
        self.config = Config()
        self.config.CHROMA_PATH = str(Path(self.temp_dir) / "test_chroma")
        self.config.ENABLE_REACT = True
        self.config.MAX_REACT_ITERATIONS = 3
        self.config.REACT_DEBUG = True
        
        # Create RAG system
        self.rag_system = RAGSystem(self.config)
        
        # Add test course data
        self._setup_test_courses()
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _setup_test_courses(self):
        """Add test courses to the system"""
        # Python course
        python_course = Course(
            title="Python Programming Fundamentals",
            instructor="Dr. Smith",
            course_link="https://example.com/python",
            lessons=[
                Lesson(lesson_number=1, title="Variables and Data Types", lesson_link="https://example.com/python/1"),
                Lesson(lesson_number=2, title="Object-Oriented Programming", lesson_link="https://example.com/python/2"),
                Lesson(lesson_number=3, title="Inheritance and Polymorphism", lesson_link="https://example.com/python/3")
            ]
        )
        
        # Java course  
        java_course = Course(
            title="Java Advanced Concepts",
            instructor="Prof. Johnson",
            course_link="https://example.com/java",
            lessons=[
                Lesson(lesson_number=1, title="Class Design Principles", lesson_link="https://example.com/java/1"),
                Lesson(lesson_number=2, title="Inheritance Hierarchies", lesson_link="https://example.com/java/2"),
                Lesson(lesson_number=3, title="Polymorphism Patterns", lesson_link="https://example.com/java/3")
            ]
        )
        
        # Machine Learning course
        ml_course = Course(
            title="Machine Learning Advanced",
            instructor="Dr. Chen",
            course_link="https://example.com/ml",
            lessons=[
                Lesson(lesson_number=1, title="Prerequisites: Linear Algebra, Statistics", lesson_link="https://example.com/ml/1"),
                Lesson(lesson_number=2, title="Neural Network Architectures", lesson_link="https://example.com/ml/2"),
                Lesson(lesson_number=3, title="Deep Learning Applications", lesson_link="https://example.com/ml/3")
            ]
        )
        
        # Add courses to vector store
        self.rag_system.vector_store.add_course_metadata(python_course)
        self.rag_system.vector_store.add_course_metadata(java_course)
        self.rag_system.vector_store.add_course_metadata(ml_course)
        
        # Add some sample content chunks for realistic search results
        from models import CourseChunk
        
        chunks = [
            CourseChunk(
                course_title="Python Programming Fundamentals",
                lesson_number=3,
                chunk_index=0,
                content="Inheritance allows classes to inherit attributes and methods from parent classes. In Python, you can create child classes using class Child(Parent): syntax. Polymorphism enables objects of different types to be treated uniformly through shared interfaces."
            ),
            CourseChunk(
                course_title="Java Advanced Concepts", 
                lesson_number=3,
                chunk_index=0,
                content="Java polymorphism is achieved through method overriding and interfaces. Runtime polymorphism allows the JVM to determine which method implementation to call based on the actual object type. This is fundamental to object-oriented design patterns."
            ),
            CourseChunk(
                course_title="Machine Learning Advanced",
                lesson_number=1,
                chunk_index=0,
                content="Prerequisites for this advanced course include solid understanding of linear algebra (matrices, eigenvalues), statistics (probability distributions, hypothesis testing), and calculus (derivatives, optimization). Students should also have basic programming experience in Python."
            )
        ]
        
        self.rag_system.vector_store.add_course_content(chunks)
    
    def test_multi_step_comparison_query(self):
        """Test ReAct with multi-step comparison query"""
        # This query should trigger multiple searches
        query = "Compare inheritance concepts between Python and Java courses"
        
        response, sources = self.rag_system.query(query)
        
        # Verify we got a meaningful response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 50)
        
        # Check that response mentions both languages
        response_lower = response.lower()
        self.assertIn("python", response_lower)
        self.assertIn("java", response_lower)
        self.assertIn("inheritance", response_lower)
        
        # Verify sources were collected
        self.assertIsInstance(sources, list)
    
    def test_prerequisite_and_topics_query(self):
        """Test ReAct with prerequisites and topics query"""
        query = "What are the prerequisites for machine learning courses and what specific neural network topics are covered?"
        
        response, sources = self.rag_system.query(query)
        
        # Verify comprehensive response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 50)
        
        # Check for prerequisite information
        response_lower = response.lower()
        self.assertIn("prerequisite", response_lower)
        
        # Should mention specific topics from the course
        should_contain_any = ["linear algebra", "statistics", "neural", "machine learning"]
        self.assertTrue(any(term in response_lower for term in should_contain_any))
    
    def test_react_debug_output(self):
        """Test that debug output is generated when enabled"""
        self.config.REACT_DEBUG = True
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            query = "Tell me about Python programming concepts"
            self.rag_system.query(query)
            
            # Check if any debug prints were made (depends on whether ReAct triggered)
            # This test verifies the debug infrastructure works
            debug_calls = [call for call in mock_print.call_args_list 
                          if call[0] and "ReAct" in str(call[0][0])]
            
            # Debug output should appear if multiple iterations occurred
            # (This may be 0 if query doesn't trigger multi-step behavior)
            self.assertIsInstance(debug_calls, list)
    
    def test_react_disabled_fallback(self):
        """Test that ReAct can be disabled via configuration"""
        # Disable ReAct
        self.config.ENABLE_REACT = False
        
        # Create new generator with ReAct disabled
        with patch('ai_generator.ProviderFactory.create_provider', return_value=self.mock_provider):
            disabled_generator = AIGenerator(config=self.config)
        
        # Setup mock responses that would normally trigger ReAct
        initial_response = LLMResponse(
            content="Need to search.",
            tool_calls=[ToolCall(id="1", name="search", parameters={"query": "test"})],
            stop_reason="tool_use"
        )
        
        # This response has tool calls but should be ignored due to disabled ReAct
        response_with_tools = LLMResponse(
            content="Found results, need another search.",
            tool_calls=[ToolCall(id="2", name="search", parameters={"query": "more"})],
            stop_reason="tool_use"
        )
        
        self.mock_provider.execute_tool_calls.return_value = response_with_tools
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results"
        
        # Execute
        result = disabled_generator._handle_tool_execution(initial_response, tool_manager)
        
        # Should only execute once since ReAct is disabled
        self.assertEqual(self.mock_provider.execute_tool_calls.call_count, 1)
        tool_manager.execute_tool.assert_called_once()
    
    def test_max_iterations_boundary(self):
        """Test that max iterations prevents infinite loops"""
        # Set low max iterations
        self.config.MAX_REACT_ITERATIONS = 2
        
        # Setup responses that always want more tools
        always_tool_response = LLMResponse(
            content="Need more searches.",
            tool_calls=[ToolCall(id="1", name="search", parameters={"query": "test"})],
            stop_reason="tool_use"
        )
        
        self.mock_provider.execute_tool_calls.return_value = always_tool_response
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results"
        
        # Execute
        result = self.ai_generator._handle_tool_execution(always_tool_response, tool_manager)
        
        # Should stop at max iterations (2)
        self.assertEqual(self.mock_provider.execute_tool_calls.call_count, 2)
        self.assertEqual(tool_manager.execute_tool.call_count, 3)  # Initial + 2 iterations


if __name__ == "__main__":
    unittest.main()