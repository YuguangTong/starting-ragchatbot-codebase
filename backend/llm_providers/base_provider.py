"""LLM provider abstraction layer for supporting multiple AI providers."""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from LLM providers"""
    content: str
    tool_calls: List[Dict[str, Any]] = None
    stop_reason: str = "end_turn"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ToolCall:
    """Standardized tool call representation"""
    id: str
    name: str
    parameters: Dict[str, Any]


@dataclass
class ToolResult:
    """Standardized tool result representation"""
    tool_call_id: str
    content: str
    is_error: bool = False


class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_params = self._get_base_params()

    @abstractmethod
    def _get_base_params(self) -> Dict[str, Any]:
        """Get provider-specific base parameters"""
        raise NotImplementedError

    @abstractmethod
    def generate_response(self,
                         query: str,
                         system_prompt: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """
        Generate response from the LLM.

        Args:
            query: The user's question or request
            system_prompt: System prompt to set behavior
            conversation_history: Previous messages for context
            tools: Available tools in standardized format

        Returns:
            LLMResponse with content and metadata
        """
        raise NotImplementedError

    @abstractmethod
    def execute_tool_calls(self,
                          initial_response: LLMResponse,
                          tool_results: List[ToolResult],
                          system_prompt: str,
                          conversation_history: Optional[str] = None) -> LLMResponse:
        """
        Execute tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool calls
            tool_results: Results from executing the tools
            system_prompt: System prompt to maintain context
            conversation_history: Previous conversation context

        Returns:
            Final LLMResponse after tool execution
        """
        raise NotImplementedError

    @abstractmethod
    def convert_tools_to_provider_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert universal tool format to provider-specific format"""
        raise NotImplementedError

    @abstractmethod
    def extract_tool_calls(self, response: Any) -> List[ToolCall]:
        """Extract tool calls from provider-specific response format"""
        raise NotImplementedError

    def get_provider_name(self) -> str:
        """Return the name of this provider"""
        return self.__class__.__name__.replace('Provider', '').lower()