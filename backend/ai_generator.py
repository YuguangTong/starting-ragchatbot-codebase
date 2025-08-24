from typing import List, Optional, Dict, Any
from llm_providers.provider_factory import ProviderFactory
from llm_providers.base_provider import LLMProvider, ToolResult

class AIGenerator:
    """Handles interactions with multiple LLM providers for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, 
                 provider_type: str = "claude",
                 anthropic_api_key: str = "",
                 anthropic_model: str = "claude-sonnet-4-20250514",
                 google_api_key: str = "",
                 gemini_model: str = "gemini-1.5-flash"):
        """
        Initialize AIGenerator with specified provider.
        
        Args:
            provider_type: Type of LLM provider ("claude", "gemini", "random")
            anthropic_api_key: Anthropic API key
            anthropic_model: Claude model to use
            google_api_key: Google API key  
            gemini_model: Gemini model to use
        """
        self.provider: LLMProvider = ProviderFactory.create_provider(
            provider_type=provider_type,
            anthropic_api_key=anthropic_api_key,
            anthropic_model=anthropic_model,
            google_api_key=google_api_key,
            gemini_model=gemini_model
        )
        self.provider_type = provider_type
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Get response from provider
        response = self.provider.generate_response(
            query=query,
            system_prompt=self.SYSTEM_PROMPT,
            conversation_history=conversation_history,
            tools=tools
        )
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, tool_manager)
        
        # Return direct response
        return response.content
    
    def _handle_tool_execution(self, initial_response, tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The LLMResponse containing tool calls
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Execute all tool calls and collect results
        tool_results = []
        for tool_call in initial_response.tool_calls:
            tool_result_content = tool_manager.execute_tool(
                tool_call.name, 
                **tool_call.parameters
            )
            
            tool_results.append(ToolResult(
                tool_call_id=tool_call.id,
                content=tool_result_content
            ))
        
        # Get final response from provider
        final_response = self.provider.execute_tool_calls(
            initial_response=initial_response,
            tool_results=tool_results,
            system_prompt=self.SYSTEM_PROMPT
        )
        
        return final_response.content