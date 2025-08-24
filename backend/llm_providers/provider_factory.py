"""Factory for creating LLM provider instances."""
import random
from typing import Optional

from .base_provider import LLMProvider
from .claude_provider import ClaudeProvider
from .gemini_provider import GeminiProvider


class ProviderFactory:
    """Factory class for creating LLM provider instances"""
    
    @staticmethod
    def create_provider(provider_type: str,
                       anthropic_api_key: str = "",
                       anthropic_model: str = "claude-sonnet-4-20250514",
                       google_api_key: str = "",
                       gemini_model: str = "gemini-1.5-flash") -> Optional[LLMProvider]:
        """
        Create an LLM provider instance based on the provider type.
        
        Args:
            provider_type: Type of provider ("claude", "gemini", or "random")
            anthropic_api_key: Anthropic API key
            anthropic_model: Claude model to use
            google_api_key: Google API key
            gemini_model: Gemini model to use
            
        Returns:
            LLMProvider instance or None if invalid configuration
        """
        
        if provider_type.lower() == "random":
            # Randomly choose between available providers
            available_providers = []
            if anthropic_api_key:
                available_providers.append("claude")
            if google_api_key:
                available_providers.append("gemini")
            
            if not available_providers:
                raise ValueError("No API keys provided for random provider selection")
            
            provider_type = random.choice(available_providers)
        
        if provider_type.lower() == "claude":
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required for Claude provider")
            return ClaudeProvider(anthropic_api_key, anthropic_model)
        
        elif provider_type.lower() == "gemini":
            if not google_api_key:
                raise ValueError("Google API key is required for Gemini provider")
            return GeminiProvider(google_api_key, gemini_model)
        
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_available_providers(anthropic_api_key: str = "", google_api_key: str = "") -> list:
        """
        Get list of available providers based on provided API keys.
        
        Args:
            anthropic_api_key: Anthropic API key
            google_api_key: Google API key
            
        Returns:
            List of available provider names
        """
        providers = []
        if anthropic_api_key:
            providers.append("claude")
        if google_api_key:
            providers.append("gemini")
        if len(providers) > 1:
            providers.append("random")
        
        return providers