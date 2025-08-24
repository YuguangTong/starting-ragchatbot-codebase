"""Anthropic Claude LLM provider implementation."""

from typing import Any, Dict, List, Optional

import anthropic

from .base_provider import LLMProvider, LLMResponse, ToolCall, ToolResult


class ClaudeProvider(LLMProvider):
    """Anthropic Claude LLM provider with function calling support"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)

    def _get_base_params(self) -> Dict[str, Any]:
        """Get Claude-specific base parameters"""
        return {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        system_prompt: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate response from Claude"""

        # Build system content with conversation history
        system_content = (
            f"{system_prompt}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else system_prompt
        )

        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        try:
            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Extract content
            content = ""
            if response.content and len(response.content) > 0:
                content = response.content[0].text

            # Extract tool calls
            tool_calls = []
            if response.stop_reason == "tool_use":
                tool_calls = self.extract_tool_calls(response)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                stop_reason=response.stop_reason,
                metadata={
                    "model": self.model,
                    "usage": response.usage.__dict__ if response.usage else {},
                },
            )

        except Exception as e:
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                stop_reason="error",
                metadata={"error": str(e)},
            )

    def execute_tool_calls(
        self,
        initial_response: LLMResponse,
        tool_results: List[ToolResult],
        system_prompt: str,
        conversation_history: Optional[str] = None,
    ) -> LLMResponse:
        """Execute tool calls and get follow-up response from Claude"""

        # Build system content
        system_content = (
            f"{system_prompt}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else system_prompt
        )

        # Build messages conversation
        messages = []

        # Add AI's tool use response - reconstruct from initial_response
        assistant_content = []
        if initial_response.content:
            assistant_content.append({"type": "text", "text": initial_response.content})

        # Add tool calls to assistant message
        for tool_call in initial_response.tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.parameters,
                }
            )

        messages.append({"role": "assistant", "content": assistant_content})

        # Add tool results as user message
        tool_result_content = []
        for result in tool_results:
            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                }
            )

        messages.append({"role": "user", "content": tool_result_content})

        # Prepare final API call
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        try:
            # Get final response
            final_response = self.client.messages.create(**final_params)

            content = ""
            if final_response.content and len(final_response.content) > 0:
                content = final_response.content[0].text

            return LLMResponse(
                content=content,
                stop_reason=final_response.stop_reason,
                metadata={
                    "model": self.model,
                    "usage": final_response.usage.__dict__
                    if final_response.usage
                    else {},
                },
            )

        except Exception as e:
            return LLMResponse(
                content=f"Error executing tool calls: {str(e)}",
                stop_reason="error",
                metadata={"error": str(e)},
            )

    def convert_tools_to_provider_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert universal tool format to Claude tool format (already compatible)"""
        return tools

    def extract_tool_calls(self, response: Any) -> List[ToolCall]:
        """Extract tool calls from Claude response format"""
        tool_calls = []

        if hasattr(response, "content"):
            for content_block in response.content:
                if hasattr(content_block, "type") and content_block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=content_block.id,
                            name=content_block.name,
                            parameters=content_block.input,
                        )
                    )

        return tool_calls
