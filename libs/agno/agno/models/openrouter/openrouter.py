from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from agno.exceptions import ModelAuthenticationError
from agno.models.openai.like import OpenAILike
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput


@dataclass
class OpenRouter(OpenAILike):
    """
    A class for using models hosted on OpenRouter.

    Attributes:
        id (str): The model id. Defaults to "gpt-4o".
        name (str): The model name. Defaults to "OpenRouter".
        provider (str): The provider name. Defaults to "OpenRouter".
        api_key (Optional[str]): The API key.
        base_url (str): The base URL. Defaults to "https://openrouter.ai/api/v1".
        max_tokens (int): The maximum number of tokens. Defaults to 1024.
        fallback_models (Optional[List[str]]): List of fallback model IDs to use if the primary model
            fails due to rate limits, timeouts, or unavailability. OpenRouter will automatically try
            these models in order. Example: ["anthropic/claude-sonnet-4", "deepseek/deepseek-r1"]
    """

    id: str = "gpt-4o"
    name: str = "OpenRouter"
    provider: str = "OpenRouter"

    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 1024
    models: Optional[List[str]] = None  # Dynamic model routing https://openrouter.ai/docs/features/model-routing

    def _get_client_params(self) -> Dict[str, Any]:
        """
        Returns client parameters for API requests, checking for OPENROUTER_API_KEY.

        Returns:
            Dict[str, Any]: A dictionary of client parameters for API requests.
        """
        # Fetch API key from env if not already set
        if not self.api_key:
            self.api_key = getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ModelAuthenticationError(
                    message="OPENROUTER_API_KEY not set. Please set the OPENROUTER_API_KEY environment variable.",
                    model_name=self.name,
                )

        return super()._get_client_params()

    def get_request_params(
        self,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_response: Optional[RunOutput] = None,
    ) -> Dict[str, Any]:
        """
        Returns keyword arguments for API requests, including fallback models configuration.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        # Get base request params from parent class
        request_params = super().get_request_params(
            response_format=response_format, tools=tools, tool_choice=tool_choice, run_response=run_response
        )

        # Add fallback models to extra_body if specified
        if self.models:
            # Get existing extra_body or create new dict
            extra_body = request_params.get("extra_body") or {}

            # Merge fallback models into extra_body
            extra_body["models"] = self.models

            # Update request params
            request_params["extra_body"] = extra_body

        return request_params

    @staticmethod
    def _get_model_extra(data: Any) -> Dict[str, Any]:
        extra = getattr(data, "model_extra", None)
        return extra if isinstance(extra, dict) else {}

    @classmethod
    def _get_usage_metadata(cls, usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}

        usage_data: Dict[str, Any] = {}
        try:
            usage_data = usage.model_dump()
        except Exception:
            try:
                usage_data = dict(usage)
            except Exception:
                usage_data = {}

        usage_extra = cls._get_model_extra(usage)
        if usage_extra:
            usage_data.update(usage_extra)
        return usage_data

    @classmethod
    def _get_choice_metadata(cls, choice: Any) -> Dict[str, Any]:
        choice_data: Dict[str, Any] = {}
        try:
            choice_data = choice.model_dump()
        except Exception:
            try:
                choice_data = dict(choice)
            except Exception:
                choice_data = {}

        choice_extra = cls._get_model_extra(choice)
        if choice_extra:
            choice_data.update(choice_extra)
        return choice_data

    def _apply_openrouter_metadata(
        self,
        model_response: ModelResponse,
        response: Union[ChatCompletion, ChatCompletionChunk],
        *,
        include_choices: bool = True,
    ) -> None:
        if model_response.provider_data is None:
            model_response.provider_data = {}

        provider_data = model_response.provider_data
        response_extra = self._get_model_extra(response)

        if getattr(response, "model", None):
            provider_data["model"] = response.model
        if getattr(response, "created", None):
            provider_data["created"] = response.created
        if getattr(response, "object", None):
            provider_data["object"] = response.object
        if "provider" in response_extra:
            provider_data["provider"] = response_extra["provider"]

        usage_data = self._get_usage_metadata(getattr(response, "usage", None))
        if usage_data:
            provider_data["usage"] = usage_data
            if model_response.response_usage is not None:
                total_cost = usage_data.get("total_cost")
                if total_cost is not None:
                    model_response.response_usage.cost = total_cost

        if include_choices and getattr(response, "choices", None):
            provider_data["choices"] = [self._get_choice_metadata(choice) for choice in response.choices]
            choice_extra = self._get_model_extra(response.choices[0])
            finish_reason = getattr(response.choices[0], "finish_reason", None) or choice_extra.get("finish_reason")
            native_finish_reason = choice_extra.get("native_finish_reason")
            if finish_reason is not None:
                provider_data["finish_reason"] = finish_reason
            if native_finish_reason is not None:
                provider_data["native_finish_reason"] = native_finish_reason

    def _parse_provider_response(
        self,
        response: ChatCompletion,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> ModelResponse:
        model_response = super()._parse_provider_response(response, response_format=response_format)
        self._apply_openrouter_metadata(model_response, response)
        return model_response

    def _parse_provider_response_delta(self, response_delta: ChatCompletionChunk) -> ModelResponse:
        model_response = super()._parse_provider_response_delta(response_delta)
        self._apply_openrouter_metadata(model_response, response_delta)
        return model_response
