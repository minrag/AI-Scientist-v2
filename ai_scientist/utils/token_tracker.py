from functools import wraps
from typing import Dict, Optional, List
import tiktoken
from collections import defaultdict
import asyncio
from datetime import datetime
import logging


class TokenTracker:
    def __init__(self):
        """
        Token counts for prompt, completion, reasoning, and cached.
        Reasoning tokens are included in completion tokens.
        Cached tokens are included in prompt tokens.
        Also tracks prompts, responses, and timestamps.
        We assume we get these from the LLM response, and we don't count
        the tokens by ourselves.
        """
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)

        # Simplified model price mapping — keys use canonical model names without date suffixes
        self.MODEL_PRICES = {
            "gpt-4o": {
                "prompt": 2.5 / 1000000,
                "cached": 1.25 / 1000000,
                "completion": 10 / 1000000,
            },
            "gpt-4o-mini": {
                "prompt": 0.15 / 1000000,
                "cached": 0.075 / 1000000,
                "completion": 0.6 / 1000000,
            },
            "o1": {
                "prompt": 15 / 1000000,
                "cached": 7.5 / 1000000,
                "completion": 60 / 1000000,
            },
            "o1-preview": {
                "prompt": 15 / 1000000,
                "cached": 7.5 / 1000000,
                "completion": 60 / 1000000,
            },
            "o3-mini": {
                "prompt": 1.1 / 1000000,
                "cached": 0.55 / 1000000,
                "completion": 4.4 / 1000000,
            },
        }

    def add_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cached_tokens: int,
    ):
        m = self._normalize_model_name(model)
        self.token_counts[m]["prompt"] += prompt_tokens
        self.token_counts[m]["completion"] += completion_tokens
        self.token_counts[m]["reasoning"] += reasoning_tokens
        self.token_counts[m]["cached"] += cached_tokens

    def add_interaction(
        self,
        model: str,
        system_message: str,
        prompt: str,
        response: str,
        timestamp: datetime,
    ):
        """Record a single interaction with the model."""
        m = self._normalize_model_name(model)
        self.interactions[m].append(
            {
                "system_message": system_message,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
            }
        )

    def get_interactions(self, model: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get all interactions, optionally filtered by model."""
        if model:
            m = self._normalize_model_name(model)
            return {m: self.interactions[m]}
        return dict(self.interactions)

    def reset(self):
        """Reset all token counts and interactions."""
        self.token_counts = defaultdict(
            lambda: {"prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
        )
        self.interactions = defaultdict(list)
        # self._encoders = {}

    def calculate_cost(self, model: str) -> float:
        """Calculate the cost for a specific model based on token usage."""
        m = self._normalize_model_name(model)
        if m not in self.MODEL_PRICES:
            logging.warning(f"Price information not available for model {m}")
            return 0.0

        prices = self.MODEL_PRICES[m]
        tokens = self.token_counts[m]

        # Calculate cost for prompt and completion tokens
        if "cached" in prices:
            prompt_cost = (tokens["prompt"] - tokens["cached"]) * prices["prompt"]
            cached_cost = tokens["cached"] * prices["cached"]
        else:
            prompt_cost = tokens["prompt"] * prices["prompt"]
            cached_cost = 0
        completion_cost = tokens["completion"] * prices["completion"]

        return prompt_cost + cached_cost + completion_cost

    def get_summary(self) -> Dict[str, Dict[str, int]]:
        # return dict(self.token_counts)
        """Get summary of token usage and costs for all models."""
        summary = {}
        for model, tokens in self.token_counts.items():
            summary[model] = {
                "tokens": tokens.copy(),
                "cost (USD)": self.calculate_cost(model),
            }
        return summary

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model names by removing date/version suffixes and mapping to canonical keys.

        Examples:
        - 'gpt-4o-2024-11-20' -> 'gpt-4o'
        - 'gpt-4o-mini-2024-07-18' -> 'gpt-4o-mini'
        - 'o1-2024-12-17' -> 'o1'
        - 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0' -> 'claude-3-5-sonnet'
        """
        if not model:
            return model
        # normalize by removing common date/version suffixes
        m = model.lower()
        # strip trailing date patterns like -YYYY-MM-DD or -YYYYMMDD
        m = re.sub(r"-\d{4}-\d{2}-\d{2}", "", m)
        m = re.sub(r"-\d{8}", "", m)
        # drop anything after colon (ollama-style tags or other qualifiers)
        if ":" in m:
            m = m.split(":")[0]
        return m


# Global token tracker instance
token_tracker = TokenTracker()


def track_token_usage(func):
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

        if hasattr(result, "usage") and result.usage.completion_tokens_details is not None:
            token_tracker.add_tokens(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.completion_tokens_details.reasoning_tokens,
                (
                    result.usage.prompt_tokens_details.cached_tokens
                    if hasattr(result.usage, "prompt_tokens_details")
                    else 0
                ),
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[
                    0
                ].message.content,  # Assumes response is in content field
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

        if hasattr(result, "usage") and result.usage.completion_tokens_details is not None:
            token_tracker.add_tokens(
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                result.usage.completion_tokens_details.reasoning_tokens,
                (
                    result.usage.prompt_tokens_details.cached_tokens
                    if hasattr(result.usage, "prompt_tokens_details")
                    else 0
                ),
            )
            # Add interaction details
            token_tracker.add_interaction(
                model,
                system_message,
                prompt,
                result.choices[
                    0
                ].message.content,  # Assumes response is in content field
                timestamp,
            )
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
