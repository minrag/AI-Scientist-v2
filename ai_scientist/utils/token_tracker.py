"""Token tracking utilities for LLM API calls.

This module provides token tracking functionality without price calculations.
Only token consumption is recorded.
"""

from functools import wraps
from typing import Dict, Optional, List
from collections import defaultdict
import asyncio
from datetime import datetime
import logging


class TokenTracker:
    """Tracker for token usage across LLM API calls.

    Tracks token counts for prompt, completion, reasoning, and cached tokens.
    Also records interaction history (prompts, responses, timestamps).
    Does NOT calculate costs/pricing.
    """

    def __init__(self):
        """Initialize the token tracker."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        """Add token counts for a single API call.

        Args:
            model: Model identifier
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            reasoning_tokens: Number of reasoning tokens (subset of completion)
            cached_tokens: Number of cached tokens (subset of prompt)
        """
        self.token_counts[model]["prompt"] += prompt_tokens
        self.token_counts[model]["completion"] += completion_tokens
        self.token_counts[model]["reasoning"] += reasoning_tokens
        self.token_counts[model]["cached"] += cached_tokens

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model.

        Args:
            model: Model identifier
            system_message: System message used
            prompt: User prompt
            response: Model response
            timestamp: When the interaction occurred
        """
        self.interactions[model].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model.

        Args:
            model: Optional model to filter by

        Returns:
            Dictionary of interactions by model
        """
        if model:
            return {model: self.interactions[model]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of token usage for all models.

        Returns:
            Dictionary with token counts per model
        """
        summary = {}
        for model, tokens in self.token_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
            }
        return summary


# Global token tracker instance
token_tracker = TokenTracker()


def track_token_usage(func):
    """Decorator to track token usage for LLM API calls.

    Wraps async or sync functions that return API response objects
    with usage information.
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )

        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        result = await func(*args, **kwargs)
        model = result.model
        timestamp = result.created

        if hasattr(result, "usage") and result.usage:
            # Get token counts with safe defaults
            prompt_tokens = result.usage.prompt_tokens if hasattr(result.usage, "prompt_tokens") else 0
            completion_tokens = result.usage.completion_tokens if hasattr(result.usage, "completion_tokens") else 0

            # Get reasoning tokens if available
            reasoning_tokens = 0
            if hasattr(result.usage, "completion_tokens_details") and result.usage.completion_tokens_details:
                reasoning_tokens = result.usage.completion_tokens_details.reasoning_tokens

            # Get cached tokens if available
            cached_tokens = 0
            if hasattr(result.usage, "prompt_tokens_details") and result.usage.prompt_tokens_details:
                cached_tokens = result.usage.prompt_tokens_details.cached_tokens

            token_tracker.add_tokens(
                model,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[0].message.content if result.choices else "",
                timestamp,
            )
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        prompt = kwargs.get("prompt")
        system_message = kwargs.get("system_message")
        if not prompt and not system_message:
            raise ValueError(
                "Either 'prompt' or 'system_message' must be provided for token tracking"
            )
        result = func(*args, **kwargs)
        model = result.model
        timestamp = result.created
        logging.info("args: ", args)
        logging.info("kwargs: ", kwargs)

        if hasattr(result, "usage") and result.usage:
            # Get token counts with safe defaults
            prompt_tokens = result.usage.prompt_tokens if hasattr(result.usage, "prompt_tokens") else 0
            completion_tokens = result.usage.completion_tokens if hasattr(result.usage, "completion_tokens") else 0

            # Get reasoning tokens if available
            reasoning_tokens = 0
            if hasattr(result.usage, "completion_tokens_details") and result.usage.completion_tokens_details:
                reasoning_tokens = result.usage.completion_tokens_details.reasoning_tokens

            # Get cached tokens if available
            cached_tokens = 0
            if hasattr(result.usage, "prompt_tokens_details") and result.usage.prompt_tokens_details:
                cached_tokens = result.usage.prompt_tokens_details.cached_tokens

            token_tracker.add_tokens(
                model,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[0].message.content if result.choices else "",
                timestamp,
            )
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
