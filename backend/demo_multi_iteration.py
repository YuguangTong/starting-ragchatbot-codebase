"""Demo script to show true multi-iteration ReAct behavior"""

from unittest.mock import Mock

from ai_generator import AIGenerator
from config import Config
from llm_providers.base_provider import LLMResponse, ToolCall


class MockProvider:
    """Mock provider that demonstrates multi-iteration behavior"""

    def __init__(self):
        self.call_count = 0
        self.responses = [
            # First iteration: AI decides to search for MCP
            LLMResponse(
                content="I found information about MCP. Now I need to search for Chroma.",
                tool_calls=[
                    ToolCall(
                        id="2",
                        name="search_course_content",
                        parameters={"query": "Chroma vector database"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Second iteration: AI decides to search for Computer Use
            LLMResponse(
                content="Good, I have MCP and Chroma info. Now I need Computer Use workflows.",
                tool_calls=[
                    ToolCall(
                        id="3",
                        name="search_course_content",
                        parameters={"query": "computer use workflow automation"},
                    )
                ],
                stop_reason="tool_use",
            ),
            # Final iteration: AI synthesizes all information
            LLMResponse(
                content="Based on all three searches, here's how these technologies work together: MCP provides the protocol layer, Chroma handles vector storage and retrieval, and Computer Use enables automated workflows. Together they create a complete AI application stack.",
                stop_reason="end_turn",
            ),
        ]

    def execute_tool_calls(
        self, initial_response, tool_results, system_prompt, conversation_history=None
    ):
        """Return next response in sequence"""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        else:
            # Fallback
            return LLMResponse(content="No more responses", stop_reason="end_turn")


def demo_multi_iteration():
    """Demonstrate true multi-iteration ReAct behavior"""
    print("🚀 DEMONSTRATING MULTI-ITERATION REACT")
    print("=" * 60)

    # Setup config with debug
    config = Config()
    config.REACT_DEBUG = True
    config.ENABLE_REACT = True
    config.MAX_REACT_ITERATIONS = 5

    # Create AI generator with mock provider
    from unittest.mock import patch

    with patch(
        "ai_generator.ProviderFactory.create_provider", return_value=MockProvider()
    ):
        ai_generator = AIGenerator(config=config)

    # Mock tool manager
    tool_manager = Mock()
    tool_manager.execute_tool.side_effect = [
        "MCP search results: MCP is a protocol for connecting AI models to tools...",
        "Chroma search results: Chroma is a vector database for embeddings...",
        "Computer Use search results: Computer Use enables automated workflows...",
    ]

    # Initial response that starts the ReAct chain
    initial_response = LLMResponse(
        content="I'll search for MCP concepts first.",
        tool_calls=[
            ToolCall(
                id="1",
                name="search_course_content",
                parameters={"query": "MCP tools protocol"},
            )
        ],
        stop_reason="tool_use",
    )

    print("Starting ReAct chain with initial search for MCP...")
    print()

    # Execute the ReAct loop
    result = ai_generator._handle_tool_execution(initial_response, tool_manager)

    print()
    print(f"Final result: {result}")
    print()
    print(f"Total tool executions: {tool_manager.execute_tool.call_count}")
    print(f"Total provider calls: {ai_generator.provider.call_count}")

    # Verify we got multiple iterations
    if ai_generator.provider.call_count >= 2:
        print("✅ SUCCESS: Multi-iteration ReAct behavior demonstrated!")
        print(f"   - {ai_generator.provider.call_count} ReAct iterations")
        print(f"   - {tool_manager.execute_tool.call_count} total tool calls")
    else:
        print("❌ Only single iteration occurred")

    print("=" * 60)


if __name__ == "__main__":
    demo_multi_iteration()
