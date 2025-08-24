"""Google Gemini LLM provider implementation."""

from typing import Any, Dict, List, Optional

import google.generativeai as genai

from .base_provider import LLMProvider, LLMResponse, ToolCall, ToolResult


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider with function calling support"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def _get_base_params(self) -> Dict[str, Any]:
        """Get Gemini-specific base parameters"""
        return {
            "model": self.model,
            "temperature": 0,
            "max_output_tokens": 800,
        }

    def generate_response(
        self,
        query: str,
        system_prompt: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Generate response from Gemini"""

        # Build the full prompt with system context
        full_prompt = system_prompt
        if conversation_history:
            full_prompt += f"\n\nPrevious conversation:\n{conversation_history}"
        full_prompt += f"\n\nUser: {query}"

        # Convert tools to Gemini format if provided
        gemini_tools = None
        if tools:
            gemini_tools = self.convert_tools_to_provider_format(tools)
            # Store tools for ReAct follow-up calls
            self._current_tools = gemini_tools

        try:
            # Generate response with or without tools
            if gemini_tools:
                response = self.client.generate_content(full_prompt, tools=gemini_tools)
            else:
                response = self.client.generate_content(full_prompt)

            # Extract content and tool calls
            content = ""
            tool_calls = []

            # Always check parts first to avoid the .text property error when function calls are present
            if hasattr(response, "parts") and response.parts:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        # Extract tool calls but don't add them to text content
                        tool_calls.extend(self.extract_tool_calls(response))

                content = "".join(text_parts)
            else:
                # Fallback to .text only if no parts (should be safe)
                try:
                    if hasattr(response, "text"):
                        content = response.text
                except ValueError:
                    # This happens when there are function calls
                    content = ""

            # Determine stop reason
            stop_reason = "end_turn"
            if tool_calls:
                stop_reason = "tool_use"

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
                metadata={"model": self.model},
            )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return LLMResponse(
                content=f"Error generating response: {str(e)}",
                stop_reason="error",
                metadata={"error": str(e), "traceback": error_details},
            )

    def execute_tool_calls(
        self,
        initial_response: LLMResponse,
        tool_results: List[ToolResult],
        system_prompt: str,
        conversation_history: Optional[str] = None,
    ) -> LLMResponse:
        """Execute tool calls and get follow-up response from Gemini"""

        # Build conversation context with tool results
        full_prompt = system_prompt
        if conversation_history:
            full_prompt += f"\n\nPrevious conversation:\n{conversation_history}"

        # Add the assistant's tool usage and results
        full_prompt += "\n\nAssistant used tools and got these results:\n"
        for result in tool_results:
            full_prompt += f"- {result.content}\n"

        full_prompt += (
            "\nPlease provide a response based on the tool results above. "
            "You can use additional tools if you need more information."
        )

        try:
            # Include tools in follow-up response to enable ReAct
            response = self.client.generate_content(
                full_prompt, 
                tools=getattr(self, '_current_tools', None)
            )

            # Extract content and potential new tool calls
            content = ""
            tool_calls = []
            
            if response.parts:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        tool_calls.extend(self.extract_tool_calls(response))
                        
                content = "".join(text_parts)

            # Determine stop reason
            stop_reason = "tool_use" if tool_calls else "end_turn"

            return LLMResponse(
                content=content, 
                tool_calls=tool_calls,
                stop_reason=stop_reason, 
                metadata={"model": self.model}
            )

        except Exception as e:
            return LLMResponse(
                content=f"Error executing tool calls: {str(e)}",
                stop_reason="error",
                metadata={"error": str(e)},
            )

    def convert_tools_to_provider_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[genai.types.Tool]:
        """Convert universal tool format to Gemini function declarations"""
        function_declarations = []

        for tool in tools:
            # Convert Anthropic tool format to Gemini function declaration
            function_declaration = genai.types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"],
            )
            function_declarations.append(function_declaration)

        # Return a single Tool with all function declarations
        return [genai.types.Tool(function_declarations=function_declarations)]

    def extract_tool_calls(self, response: Any) -> List[ToolCall]:
        """Extract tool calls from Gemini response format"""
        tool_calls = []

        if hasattr(response, "parts"):
            for i, part in enumerate(response.parts):
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{i}",  # Gemini doesn't provide IDs, so we generate them
                            name=fc.name,
                            parameters=dict(fc.args),
                        )
                    )

        return tool_calls
