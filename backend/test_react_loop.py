"""Unit tests for ReAct loop functionality in AIGenerator"""

import unittest
from unittest.mock import Mock, patch

from ai_generator import AIGenerator
from config import Config
from llm_providers.base_provider import LLMResponse, ToolCall


class TestReActLoop(unittest.TestCase):
    """Test ReAct loop behavior in AIGenerator"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.config.ENABLE_REACT = True
        self.config.MAX_REACT_ITERATIONS = 3
        self.config.REACT_DEBUG = False

        # Mock provider
        self.mock_provider = Mock()

        # Create AIGenerator with mock provider
        with patch(
            "ai_generator.ProviderFactory.create_provider",
            return_value=self.mock_provider,
        ):
            self.ai_generator = AIGenerator(config=self.config)

    def test_single_tool_execution_no_react(self):
        """Test single tool execution without ReAct loop"""
        # Setup mock responses
        initial_response = LLMResponse(
            content="I need to search for information.",
            tool_calls=[ToolCall(id="1", name="search", parameters={"query": "test"})],
            stop_reason="tool_use",
        )

        final_response = LLMResponse(
            content="Here's the answer based on search results.", stop_reason="end_turn"
        )

        self.mock_provider.execute_tool_calls.return_value = final_response

        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results content"

        # Execute
        result = self.ai_generator._handle_tool_execution(
            initial_response, tool_manager
        )

        # Verify single execution
        self.assertEqual(result, "Here's the answer based on search results.")
        self.mock_provider.execute_tool_calls.assert_called_once()
        tool_manager.execute_tool.assert_called_once_with("search", query="test")

    def test_multi_round_react_loop(self):
        """Test multi-round ReAct loop"""
        # Setup mock responses for 3 iterations
        initial_response = LLMResponse(
            content="I need to search for Python courses.",
            tool_calls=[
                ToolCall(id="1", name="search", parameters={"query": "Python"})
            ],
            stop_reason="tool_use",
        )

        # Second iteration response
        second_response = LLMResponse(
            content="Now I need to search for Java courses.",
            tool_calls=[ToolCall(id="2", name="search", parameters={"query": "Java"})],
            stop_reason="tool_use",
        )

        # Final response
        final_response = LLMResponse(
            content="Based on both searches, here's the comparison.",
            stop_reason="end_turn",
        )

        # Configure mock to return different responses on each call
        self.mock_provider.execute_tool_calls.side_effect = [
            second_response,
            final_response,
        ]

        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Python search results",
            "Java search results",
        ]

        # Execute
        result = self.ai_generator._handle_tool_execution(
            initial_response, tool_manager
        )

        # Verify multiple executions
        self.assertEqual(result, "Based on both searches, here's the comparison.")
        self.assertEqual(self.mock_provider.execute_tool_calls.call_count, 2)
        self.assertEqual(tool_manager.execute_tool.call_count, 2)

        # Verify tool calls
        expected_calls = [
            unittest.mock.call("search", query="Python"),
            unittest.mock.call("search", query="Java"),
        ]
        tool_manager.execute_tool.assert_has_calls(expected_calls)

    def test_max_iterations_safety_limit(self):
        """Test that ReAct loop respects max iterations limit"""
        # Set low max iterations for this test
        self.config.MAX_REACT_ITERATIONS = 2

        # Setup mock to always return tool_use
        tool_response = LLMResponse(
            content="Still need more searches.",
            tool_calls=[ToolCall(id="1", name="search", parameters={"query": "test"})],
            stop_reason="tool_use",
        )

        self.mock_provider.execute_tool_calls.return_value = tool_response

        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results"

        # Execute with initial response that wants to use tools
        initial_response = LLMResponse(
            content="Starting search.",
            tool_calls=[ToolCall(id="1", name="search", parameters={"query": "start"})],
            stop_reason="tool_use",
        )

        result = self.ai_generator._handle_tool_execution(
            initial_response, tool_manager
        )

        # With MAX_REACT_ITERATIONS=2, should run exactly 2 iterations
        self.assertEqual(self.mock_provider.execute_tool_calls.call_count, 2)
        self.assertEqual(
            tool_manager.execute_tool.call_count, 2
        )  # 2 tool executions total

    def test_react_disabled_fallback(self):
        """Test fallback to single execution when ReAct is disabled"""
        # Disable ReAct
        self.config.ENABLE_REACT = False

        # Setup mock response that would normally trigger ReAct
        initial_response = LLMResponse(
            content="I need to search.",
            tool_calls=[ToolCall(id="1", name="search", parameters={"query": "test"})],
            stop_reason="tool_use",
        )

        # This response would normally trigger another round
        second_response = LLMResponse(
            content="I need another search.",
            tool_calls=[ToolCall(id="2", name="search", parameters={"query": "more"})],
            stop_reason="tool_use",
        )

        self.mock_provider.execute_tool_calls.return_value = second_response

        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Search results"

        # Execute
        result = self.ai_generator._handle_tool_execution(
            initial_response, tool_manager
        )

        # Should only execute once since ReAct is disabled
        self.assertEqual(self.mock_provider.execute_tool_calls.call_count, 1)
        tool_manager.execute_tool.assert_called_once_with("search", query="test")


if __name__ == "__main__":
    unittest.main()
