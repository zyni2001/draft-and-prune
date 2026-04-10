#!/usr/bin/env python3
import base64
import json
import time
import random
import os
import requests
from typing import Optional, Tuple, Dict, Any, Union, List
from abc import ABC, abstractmethod
import openai
from google import genai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

class APIConfig:
    """Configuration class for API calls"""
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        gemini_thinking_budget: Optional[int] = None,
        gemini_thinking_level: Optional[str] = None,
        openai_compatible_extra_body: Optional[Dict[str, Any]] = None,
        openai_compatible_timeout_sec: int = 60,
        openai_compatible_max_tokens: Optional[int] = None,
        max_retries: int = 10,
        inter_test_case_delay: float = 2.0
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.gemini_thinking_budget = gemini_thinking_budget
        self.gemini_thinking_level = gemini_thinking_level
        self.openai_compatible_extra_body = openai_compatible_extra_body
        self.openai_compatible_timeout_sec = openai_compatible_timeout_sec
        self.openai_compatible_max_tokens = openai_compatible_max_tokens
        self.max_retries = max_retries
        self.inter_test_case_delay = inter_test_case_delay

class APIClient(ABC):
    """Base class for API clients"""
    def __init__(self, config: APIConfig):
        self.config = config
        self.last_call_metadata: Dict[str, Any] = {}
    
    @property
    def temperature(self) -> float:
        """Get the current temperature setting"""
        return self.config.temperature
    
    @temperature.setter
    def temperature(self, value: float):
        """Set the temperature setting"""
        self.config.temperature = value

    def _clear_last_call_metadata(self) -> None:
        """Clear metadata from the previous API call."""
        self.last_call_metadata = {}

    def _set_last_call_usage(self, usage_obj: Any) -> None:
        """Store token usage metadata from provider response objects."""
        existing_timing = self.last_call_metadata.get("timing")
        if usage_obj is None:
            self.last_call_metadata = {"usage": None}
            if existing_timing is not None:
                self.last_call_metadata["timing"] = existing_timing
            return

        self.last_call_metadata = {
            "usage": {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
        }
        if existing_timing is not None:
            self.last_call_metadata["timing"] = existing_timing

    def _set_last_call_timing(self, api_time_only_sec: float, attempt_count: int) -> None:
        """Store API-only timing for the latest call."""
        existing_usage = self.last_call_metadata.get("usage")
        self.last_call_metadata = {
            "timing": {
                "api_time_only_sec": float(api_time_only_sec),
                "attempt_count": int(attempt_count),
            }
        }
        if existing_usage is not None:
            self.last_call_metadata["usage"] = existing_usage

    def get_last_call_metadata(self) -> Dict[str, Any]:
        """Return a shallow copy of metadata for the latest API call."""
        return dict(self.last_call_metadata)
    
    @abstractmethod
    def call(self, prompt: str) -> str:
        """Make an API call with the given prompt and return the response"""
        pass

    def call_with_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Structured message/tool API surface for agent-style loops."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support structured message/tool calls."
        )

class GeminiClient(APIClient):
    """Client for Gemini API using the official google-genai SDK."""
    def __init__(self, config: APIConfig, api_key: str = None, api_keys: list = None):
        super().__init__(config)
        if api_keys and isinstance(api_keys, list):
            self.api_keys = [k for k in api_keys if k]
        else:
            self.api_keys = [api_key] if api_key else []
        self.current_key_index = 0
        self.api_key = self.api_keys[0] if self.api_keys else api_key
        self.client = None
        self.set_api_key(self.api_key)

    def set_api_key(self, api_key: Optional[str]) -> None:
        """Update API key and reconfigure SDK client."""
        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client()

    def rotate_api_key(self):
        """Rotate to the next API key if multiple keys are available"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.set_api_key(self.api_keys[self.current_key_index])
            print(f"Rotated to API key #{self.current_key_index + 1}")
            return True
        return False

    @staticmethod
    def _build_usage_obj(usage_metadata: Any) -> Any:
        """Normalize provider usage objects to a common schema."""
        return type("UsageObj", (), {
            "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
            "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
            "total_tokens": getattr(usage_metadata, "total_token_count", None),
        })()

    @staticmethod
    def _extract_text_from_new_response(response: Any) -> Optional[str]:
        """Extract only text parts from google-genai response objects.

        Avoids accessing `response.text`, which can emit warnings when the
        response includes non-text parts such as `thought_signature`.
        """
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            extracted = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    extracted.append(part_text)
            if extracted:
                return "".join(extracted)
        return None
    
    def call(self, prompt: str) -> str:
        """Call the Gemini API with error handling and retries"""
        self._clear_last_call_metadata()
        api_time_only_sec = 0.0
        attempt_count = 0

        # Try multiple times in case of errors
        for attempt in range(self.config.max_retries):
            try:
                attempt_count += 1
                api_call_start = time.time()
                generation_config: Dict[str, Any] = {
                    "temperature": self.config.temperature,
                }
                if self.config.gemini_thinking_level is not None:
                    generation_config["thinking_config"] = {
                        "thinking_level": self.config.gemini_thinking_level,
                        "include_thoughts": False,
                    }
                elif self.config.gemini_thinking_budget is not None:
                    generation_config["thinking_config"] = {
                        "thinking_budget": self.config.gemini_thinking_budget,
                        "include_thoughts": False,
                    }
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=generation_config,
                )
                api_time_only_sec += time.time() - api_call_start

                response_text = self._extract_text_from_new_response(response)

                if response_text:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    usage_metadata = getattr(response, "usage_metadata", None)
                    if usage_metadata is None:
                        self._set_last_call_usage(None)
                    else:
                        self._set_last_call_usage(self._build_usage_obj(usage_metadata))
                    return response_text

                block_reason = "Unknown"
                prompt_feedback = getattr(response, "prompt_feedback", None)
                block_reason = getattr(prompt_feedback, "block_reason", "Unknown") if prompt_feedback else "Unknown"
                print(f"Warning: Model response was empty or blocked (Attempt {attempt+1}/{self.config.max_retries}). Reason: {block_reason}")
                if block_reason != 'Unknown' and attempt < self.config.max_retries - 1:
                    print(f"Retrying due to block reason: {block_reason}")
                    time.sleep(self.config.inter_test_case_delay**(attempt+1))  # Exponential backoff
                    continue
                self._set_last_call_timing(api_time_only_sec, attempt_count)
                self._set_last_call_usage(None)
                return f"Generation failed. Reason: {block_reason}"

            except Exception as e:
                if 'api_call_start' in locals():
                    api_time_only_sec += time.time() - api_call_start
                error_str = str(e)
                print(f"Error in API call (attempt {attempt+1}/{self.config.max_retries}): {error_str}")
                
                # Check if it's a rate limit error and try rotating API keys
                if ("quota" in error_str.lower() or "rate" in error_str.lower() or "limit" in error_str.lower()):
                    if self.rotate_api_key():
                        print("Retrying with rotated API key...")
                        time.sleep(2)  # Short delay after key rotation
                        continue
                
                if attempt == self.config.max_retries - 1:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(None)
                    return f"API call failed after {self.config.max_retries} attempts: {error_str}"
                print(f"Waiting {self.config.inter_test_case_delay**(attempt+1)} seconds before retry...")
                time.sleep(self.config.inter_test_case_delay**(attempt+1))
            finally:
                # Basic rate limiting delay
                time.sleep(1.1)

        self._set_last_call_timing(api_time_only_sec, attempt_count)
        self._set_last_call_usage(None)
        return "Max retries reached for API call."

    @staticmethod
    def _normalize_tool_arguments(arguments: Any) -> Dict[str, Any]:
        """Normalize serialized tool-call arguments into a dict."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            return json.loads(arguments)
        raise ValueError("Tool arguments must be a dict or JSON string.")

    @staticmethod
    def _parse_tool_result_content(content: Any) -> Dict[str, Any]:
        """Normalize tool result payloads for Gemini function responses."""
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
            except json.JSONDecodeError:
                return {"result": content}
        if content is None:
            return {"result": None}
        return {"result": content}

    @staticmethod
    def _convert_openai_tools_to_gemini(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style function tools to Gemini declarations."""
        gemini_declarations = []
        for tool in tools:
            function_def = tool.get("function", {})
            gemini_declarations.append({
                "name": function_def.get("name"),
                "description": function_def.get("description", ""),
                "parameters": function_def.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"function_declarations": gemini_declarations}] if gemini_declarations else []

    def _convert_common_messages_to_gemini(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert common chat/tool history into Gemini contents."""
        contents: List[Dict[str, Any]] = []
        tool_name_by_id: Dict[str, str] = {}
        pending_tool_responses: List[Dict[str, Any]] = []

        def flush_pending_tool_responses() -> None:
            if pending_tool_responses:
                contents.append({"role": "user", "parts": list(pending_tool_responses)})
                pending_tool_responses.clear()

        for message in messages:
            role = message.get("role")
            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                tool_name = tool_name_by_id.get(tool_call_id, message.get("name", "tool"))
                pending_tool_responses.append({
                    "function_response": {
                        "name": tool_name,
                        "response": self._parse_tool_result_content(message.get("content")),
                    }
                })
                continue

            flush_pending_tool_responses()
            if role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": message.get("content") or ""}],
                })
                continue

            if role == "assistant":
                parts: List[Dict[str, Any]] = []
                content = message.get("content")
                if content:
                    parts.append({"text": content})
                for idx, tool_call in enumerate(message.get("tool_calls") or [], start=1):
                    function_obj = tool_call.get("function", {})
                    tool_name = function_obj.get("name")
                    tool_call_id = tool_call.get("id") or f"gemini-tool-{idx}"
                    if tool_name:
                        tool_name_by_id[tool_call_id] = tool_name
                    function_part: Dict[str, Any] = {
                        "function_call": {
                            "name": tool_name,
                            "args": self._normalize_tool_arguments(function_obj.get("arguments")),
                        }
                    }
                    if tool_call.get("thought_signature") is not None:
                        function_part["thought_signature"] = self._decode_gemini_thought_signature(
                            tool_call.get("thought_signature")
                        )
                    parts.append(function_part)
                if parts:
                    contents.append({"role": "model", "parts": parts})

        flush_pending_tool_responses()
        return contents

    @staticmethod
    def _normalize_gemini_tool_choice(tool_choice: Optional[Union[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Map common tool choice values to Gemini tool config."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            mode = tool_choice.upper()
            if mode in {"AUTO", "NONE", "ANY", "VALIDATED"}:
                return {"function_calling_config": {"mode": mode}}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function_obj = tool_choice.get("function", {})
                name = function_obj.get("name")
                if name:
                    return {
                        "function_calling_config": {
                            "mode": "ANY",
                            "allowed_function_names": [name],
                        }
                    }
            choice_type = tool_choice.get("type")
            if isinstance(choice_type, str):
                normalized = choice_type.upper()
                if normalized in {"AUTO", "NONE", "ANY", "VALIDATED"}:
                    return {"function_calling_config": {"mode": normalized}}
        return None

    @staticmethod
    def _encode_gemini_thought_signature(thought_signature: Any) -> Optional[str]:
        """Serialize Gemini thought signatures into JSON-safe strings."""
        if thought_signature is None:
            return None
        if isinstance(thought_signature, bytes):
            encoded = base64.b64encode(thought_signature).decode("ascii")
            return f"base64:{encoded}"
        if isinstance(thought_signature, str):
            return thought_signature
        return str(thought_signature)

    @staticmethod
    def _decode_gemini_thought_signature(thought_signature: Any) -> Any:
        """Restore serialized Gemini thought signatures for replay into the SDK."""
        if not isinstance(thought_signature, str):
            return thought_signature
        if not thought_signature.startswith("base64:"):
            return thought_signature
        encoded = thought_signature[len("base64:"):]
        return base64.b64decode(encoded.encode("ascii"))

    @staticmethod
    def _serialize_gemini_function_call(function_call: Any, idx: int, thought_signature: Any = None) -> Dict[str, Any]:
        """Normalize Gemini function-call objects to the common schema."""
        args = getattr(function_call, "args", None)
        if args is None and isinstance(function_call, dict):
            args = function_call.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            args = dict(args)

        serialized = {
            "id": getattr(function_call, "id", None) or f"gemini-tool-{idx}",
            "type": "function",
            "function": {
                "name": getattr(function_call, "name", None) or (function_call.get("name") if isinstance(function_call, dict) else None),
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }
        if thought_signature is not None:
            serialized["thought_signature"] = GeminiClient._encode_gemini_thought_signature(thought_signature)
        return serialized

    def call_with_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call Gemini with structured messages/tools and normalize the response."""
        self._clear_last_call_metadata()
        api_time_only_sec = 0.0
        attempt_count = 0
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                attempt_count += 1
                generation_config: Dict[str, Any] = {
                    "temperature": self.config.temperature,
                    "automatic_function_calling": {"disable": True},
                }
                if self.config.gemini_thinking_level is not None:
                    generation_config["thinking_config"] = {
                        "thinking_level": self.config.gemini_thinking_level,
                        "include_thoughts": False,
                    }
                elif self.config.gemini_thinking_budget is not None:
                    generation_config["thinking_config"] = {
                        "thinking_budget": self.config.gemini_thinking_budget,
                        "include_thoughts": False,
                    }

                if tools:
                    generation_config["tools"] = self._convert_openai_tools_to_gemini(tools)
                tool_config = self._normalize_gemini_tool_choice(tool_choice)
                if tool_config is not None:
                    generation_config["tool_config"] = tool_config

                api_call_start = time.time()
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=self._convert_common_messages_to_gemini(messages),
                    config=generation_config,
                )
                api_time_only_sec += time.time() - api_call_start

                serialized_tool_calls = []
                candidates = getattr(response, "candidates", None) or []
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        function_call = getattr(part, "function_call", None)
                        if function_call is not None:
                            serialized_tool_calls.append(
                                self._serialize_gemini_function_call(
                                    function_call,
                                    len(serialized_tool_calls) + 1,
                                    thought_signature=getattr(part, "thought_signature", None),
                                )
                            )
                if not serialized_tool_calls:
                    raw_function_calls = getattr(response, "function_calls", None) or []
                    serialized_tool_calls = [
                        self._serialize_gemini_function_call(function_call, idx + 1)
                        for idx, function_call in enumerate(raw_function_calls)
                    ]

                response_text = self._extract_text_from_new_response(response)
                finish_reason = None
                candidates = getattr(response, "candidates", None) or []
                if candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", None)
                if serialized_tool_calls and finish_reason is None:
                    finish_reason = "tool_calls"

                if response_text or serialized_tool_calls:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    usage_metadata = getattr(response, "usage_metadata", None)
                    if usage_metadata is None:
                        self._set_last_call_usage(None)
                    else:
                        self._set_last_call_usage(self._build_usage_obj(usage_metadata))
                    return {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": serialized_tool_calls,
                        "finish_reason": finish_reason,
                    }

                last_error = "Gemini response was empty."
                print(f"Warning: {last_error}")
            except Exception as e:
                if 'api_call_start' in locals():
                    api_time_only_sec += time.time() - api_call_start
                last_error = f"Error in Gemini structured API call (attempt {attempt+1}/{self.config.max_retries}): {str(e)}"
                print(last_error)
                if ("quota" in last_error.lower() or "rate" in last_error.lower() or "limit" in last_error.lower()):
                    if self.rotate_api_key():
                        print("Retrying with rotated API key...")
                        time.sleep(2)
                        continue
                if attempt == self.config.max_retries - 1:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(None)
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                        "finish_reason": "error",
                        "error": last_error,
                    }
                backoff = self.config.inter_test_case_delay ** (attempt + 1)
                print(f"Waiting {backoff} seconds before retry...")
                time.sleep(backoff)
            finally:
                time.sleep(1.1)

        self._set_last_call_timing(api_time_only_sec, attempt_count)
        self._set_last_call_usage(None)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
            "finish_reason": "error",
            "error": last_error,
        }

class GPTClient(APIClient):
    """Client for OpenAI's GPT API"""
    def call(self, prompt: str) -> str:
        """Call the GPT API with error handling and retries"""
        self._clear_last_call_metadata()
        api_time_only_sec = 0.0
        attempt_count = 0
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                attempt_count += 1
                api_call_start = time.time()
                response = openai.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    n=1,
                )
                api_time_only_sec += time.time() - api_call_start
                # Extract the content from the choices
                if response.choices and len(response.choices) > 0:
                    if response.choices[0].message:
                        self._set_last_call_timing(api_time_only_sec, attempt_count)
                        self._set_last_call_usage(getattr(response, "usage", None))
                        return response.choices[0].message.content
                    else:
                        error_message = "GPT response was empty."
                        print(f"Warning: {error_message}")
                        last_error = error_message
                        continue # Retry
                else:
                    error_message = "GPT response was empty."
                    print(f"Warning: {error_message}")
                    last_error = error_message
                    continue # Retry
            except Exception as e:
                if 'api_call_start' in locals():
                    api_time_only_sec += time.time() - api_call_start
                error_message = f"Error in API call (attempt {attempt+1}/{self.config.max_retries}): {str(e)}"
                print(error_message)
                last_error = error_message
                if attempt == self.config.max_retries - 1:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(None)
                    return f"API call failed after {self.config.max_retries} attempts. Last error: {last_error}"
                random_sleep = random.uniform(2 ** attempt, 2 ** (attempt + 1))
                print(f"Waiting {random_sleep} seconds before retry...")
                time.sleep(random_sleep) # Exponential backoff
            finally:
                time.sleep(1.1) # Rate limiting delay
        self._set_last_call_timing(api_time_only_sec, attempt_count)
        self._set_last_call_usage(None)
        return f"API call failed after {self.config.max_retries} attempts. Last error: {last_error}"

class AzureOpenAIClient(APIClient):
    """Client for Azure OpenAI API with Entra ID authentication"""
    def __init__(self, config: APIConfig, endpoint: str = None, deployment: str = None, managed_identity_client_id: str = None):
        super().__init__(config)
        self.endpoint = endpoint
        self.deployment = deployment
        self.managed_identity_client_id = managed_identity_client_id
        
        # Initialize Azure OpenAI client with Entra ID authentication
        if self.managed_identity_client_id:
            credential = DefaultAzureCredential(managed_identity_client_id=self.managed_identity_client_id)
        else:
            credential = DefaultAzureCredential()
        
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        
        # Map models to appropriate API versions
        if self.config.model_name == "gpt-4":
            api_version="2024-02-01"  # Updated from 2023-05-15
        else:
            # Default to a recent stable version for unknown models
            api_version="2024-02-01"
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )

    def _use_max_completion_tokens(self) -> bool:
        """GPT-5.* deployments require max_completion_tokens instead of max_tokens."""
        name = (self.deployment or self.config.model_name or "").lower()
        return name.startswith("gpt-5") or name.startswith("gpt5")
    
    def call(self, prompt: str) -> str:
        """Call the Azure OpenAI API with error handling and retries"""
        self._clear_last_call_metadata()
        api_time_only_sec = 0.0
        attempt_count = 0
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                request_args = {
                    "model": self.deployment,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.temperature,
                    "n": 1,
                    "stop": None,
                    "stream": False,
                }
                if self._use_max_completion_tokens():
                    request_args["max_completion_tokens"] = 2048
                else:
                    request_args["max_tokens"] = 2048

                attempt_count += 1
                api_call_start = time.time()
                response = self.client.chat.completions.create(**request_args)
                api_time_only_sec += time.time() - api_call_start
                
                # Extract the content from the choices
                if response.choices and len(response.choices) > 0:
                    if response.choices[0].message:
                        self._set_last_call_timing(api_time_only_sec, attempt_count)
                        self._set_last_call_usage(getattr(response, "usage", None))
                        return response.choices[0].message.content
                    else:
                        error_message = "Azure OpenAI response was empty."
                        print(f"Warning: {error_message}")
                        last_error = error_message
                        continue # Retry
                else:
                    error_message = "Azure OpenAI response was empty."
                    print(f"Warning: {error_message}")
                    last_error = error_message
                    continue # Retry
            except Exception as e:
                if 'api_call_start' in locals():
                    api_time_only_sec += time.time() - api_call_start
                error_message = f"Error in Azure OpenAI API call (attempt {attempt+1}/{self.config.max_retries}): {str(e)}"
                print(error_message)
                last_error = error_message
                if attempt == self.config.max_retries - 1:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(None)
                    return f"Azure OpenAI API call failed after {self.config.max_retries} attempts. Last error: {last_error}"
                random_sleep = random.uniform(2 ** attempt, 2 ** (attempt + 1))
                print(f"Waiting {random_sleep} seconds before retry...")
                time.sleep(random_sleep) # Exponential backoff
            finally:
                time.sleep(1.1) # Rate limiting delay
        self._set_last_call_timing(api_time_only_sec, attempt_count)
        self._set_last_call_usage(None)
        return f"Azure OpenAI API call failed after {self.config.max_retries} attempts. Last error: {last_error}"

class OpenAICompatibleClient(APIClient):
    """Client for OpenAI-compatible chat completion endpoints."""
    def __init__(self, config: APIConfig, api_key: str = None, base_url: str = None):
        super().__init__(config)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if isinstance(base_url, str) else base_url
        if base_url:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=api_key)

    def _is_minimax_endpoint(self) -> bool:
        return isinstance(self.base_url, str) and "api.minimax.io" in self.base_url

    def _is_bigmodel_endpoint(self) -> bool:
        return isinstance(self.base_url, str) and "open.bigmodel.cn" in self.base_url

    @staticmethod
    def _sanitize_bigmodel_temperature(temperature: float) -> float:
        """Zhipu OpenAI-compat docs: temperature must lie in (0, 1); 0 is not supported."""
        try:
            t = float(temperature)
        except (TypeError, ValueError):
            t = 0.01
        if t <= 0.0:
            return 0.01
        if t >= 1.0:
            return 0.99
        return t

    def _parse_openai_style_chat_completion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a /chat/completions JSON body to the shape used by call_with_messages."""
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-style response did not contain any choices.")
        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls_raw = message.get("tool_calls") or []
        serialized_tool_calls = []
        for tc in tool_calls_raw:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            serialized_tool_calls.append({
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                },
            })
        self._set_last_call_usage(self._build_usage_from_dict(data.get("usage")))
        return {
            "role": message.get("role", "assistant"),
            "content": message.get("content"),
            "tool_calls": serialized_tool_calls,
            "finish_reason": choice.get("finish_reason"),
        }

    def _call_bigmodel_chat_completion(self, request_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call Zhipu (bigmodel.cn) via raw HTTP; avoids SDK-only fields (e.g. n=1) and applies GLM quirks."""
        timeout = request_args.get("timeout", self.config.openai_compatible_timeout_sec)
        max_tok = getattr(self.config, "openai_compatible_max_tokens", None) or 8192
        payload: Dict[str, Any] = {
            "model": request_args["model"],
            "messages": request_args["messages"],
            "temperature": self._sanitize_bigmodel_temperature(request_args["temperature"]),
            "max_tokens": max_tok,
        }
        extra_body = request_args.get("extra_body")
        if isinstance(extra_body, dict) and extra_body:
            payload.update(extra_body)
        payload["temperature"] = self._sanitize_bigmodel_temperature(
            payload.get("temperature", request_args["temperature"])
        )
        if request_args.get("tools") is not None:
            payload["tools"] = request_args["tools"]
        if request_args.get("tool_choice") is not None:
            payload["tool_choice"] = request_args["tool_choice"]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(self._format_requests_error(response))
        data = response.json()
        return self._parse_openai_style_chat_completion(data)

    def _build_usage_from_dict(self, usage: Optional[Dict[str, Any]]) -> Any:
        usage = usage or {}
        return type("UsageObj", (), {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        })()

    @staticmethod
    def _format_requests_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = response.text
        return f"Error code: {response.status_code} - {payload}"

    def _call_minimax_chat_completion(self, request_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call MiniMax's OpenAI-style chat completions endpoint via raw HTTP.

        MiniMax's function-calling path behaves more reliably when we send the
        JSON payload directly instead of relying on SDK request shaping.
        """
        timeout = request_args.get("timeout", self.config.openai_compatible_timeout_sec)
        payload: Dict[str, Any] = {
            "model": request_args["model"],
            "messages": request_args["messages"],
        }
        if "temperature" in request_args:
            payload["temperature"] = request_args["temperature"]
        if request_args.get("tools") is not None:
            payload["tools"] = request_args["tools"]
        if request_args.get("tool_choice") is not None:
            payload["tool_choice"] = request_args["tool_choice"]
        extra_body = request_args.get("extra_body")
        if isinstance(extra_body, dict) and extra_body:
            payload.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(self._format_requests_error(response))

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("MiniMax response did not contain any choices.")

        choice = choices[0]
        message = choice.get("message") or {}
        self._set_last_call_usage(self._build_usage_from_dict(data.get("usage")))
        return {
            "role": message.get("role", "assistant"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
        }

    @staticmethod
    def _serialize_tool_call(tool_call: Any) -> Dict[str, Any]:
        """Convert SDK tool-call objects into plain dictionaries."""
        if tool_call is None:
            return {}

        function_obj = getattr(tool_call, "function", None)
        function_payload = {
            "name": getattr(function_obj, "name", None),
            "arguments": getattr(function_obj, "arguments", None),
        }

        return {
            "id": getattr(tool_call, "id", None),
            "type": getattr(tool_call, "type", "function"),
            "function": function_payload,
        }

    def call_with_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call an OpenAI-compatible chat endpoint with structured messages/tools."""
        self._clear_last_call_metadata()
        api_time_only_sec = 0.0
        attempt_count = 0
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                attempt_count += 1
                request_args: Dict[str, Any] = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "n": 1,
                    "timeout": self.config.openai_compatible_timeout_sec,
                }
                if tools is not None:
                    request_args["tools"] = tools
                if tool_choice is not None:
                    request_args["tool_choice"] = tool_choice
                if isinstance(self.config.openai_compatible_extra_body, dict) and self.config.openai_compatible_extra_body:
                    request_args["extra_body"] = self.config.openai_compatible_extra_body

                api_call_start = time.time()
                if self._is_minimax_endpoint():
                    structured_response = self._call_minimax_chat_completion(request_args)
                    response = None
                elif self._is_bigmodel_endpoint():
                    structured_response = self._call_bigmodel_chat_completion(request_args)
                    response = None
                else:
                    response = self.client.chat.completions.create(**request_args)
                    structured_response = None
                api_time_only_sec += time.time() - api_call_start

                if structured_response is not None:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    return structured_response

                if response.choices and len(response.choices) > 0 and response.choices[0].message:
                    choice = response.choices[0]
                    message = choice.message
                    serialized_tool_calls = [
                        self._serialize_tool_call(tool_call)
                        for tool_call in (getattr(message, "tool_calls", None) or [])
                    ]
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(getattr(response, "usage", None))
                    return {
                        "role": getattr(message, "role", "assistant"),
                        "content": getattr(message, "content", None),
                        "tool_calls": serialized_tool_calls,
                        "finish_reason": getattr(choice, "finish_reason", None),
                    }

                last_error = "OpenAI-compatible response was empty."
                print(f"Warning: {last_error}")

            except Exception as e:
                if 'api_call_start' in locals():
                    api_time_only_sec += time.time() - api_call_start
                last_error = f"Error in OpenAI-compatible API call (attempt {attempt+1}/{self.config.max_retries}): {str(e)}"
                print(last_error)
                if attempt == self.config.max_retries - 1:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(None)
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                        "finish_reason": "error",
                        "error": last_error,
                    }
                random_sleep = random.uniform(2 ** attempt, 2 ** (attempt + 1))
                print(f"Waiting {random_sleep} seconds before retry...")
                time.sleep(random_sleep)
            finally:
                time.sleep(1.1)

        self._set_last_call_timing(api_time_only_sec, attempt_count)
        self._set_last_call_usage(None)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
            "finish_reason": "error",
            "error": last_error,
        }

    def call(self, prompt: str) -> str:
        """Call an OpenAI-compatible chat endpoint with retries."""
        response = self.call_with_messages([{"role": "user", "content": prompt}])
        if response.get("content"):
            return response["content"]

        error_message = response.get("error")
        if error_message:
            return f"OpenAI-compatible API call failed after {self.config.max_retries} attempts. Last error: {error_message}"
        return "OpenAI-compatible response was empty."

class AnthropicClient(APIClient):
    """Client for Anthropic Messages API."""
    def __init__(self, config: APIConfig, api_key: str = None, base_url: str = None):
        super().__init__(config)
        self.api_key = api_key
        self.base_url = (base_url or "https://api.anthropic.com/v1").rstrip("/")

    def _build_usage_obj(self, usage: Dict[str, Any]):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            total_tokens = input_tokens + output_tokens
        return type("UsageObj", (), {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        })()

    def call(self, prompt: str) -> str:
        """Call Anthropic Messages API with retries."""
        self._clear_last_call_metadata()
        api_time_only_sec = 0.0
        attempt_count = 0
        last_error = None

        if not self.api_key:
            self._set_last_call_timing(api_time_only_sec, attempt_count)
            self._set_last_call_usage(None)
            return "Anthropic API key is missing."

        for attempt in range(self.config.max_retries):
            try:
                attempt_count += 1
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": self.config.model_name,
                    "max_tokens": getattr(self.config, "anthropic_max_tokens", 512),
                    "temperature": self.config.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }

                api_call_start = time.time()
                response = requests.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                api_time_only_sec += time.time() - api_call_start

                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

                body = response.json()
                content = body.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "".join(text_parts).strip()
                if text:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    usage = body.get("usage") or {}
                    self._set_last_call_usage(self._build_usage_obj(usage))
                    return text

                last_error = "Anthropic response was empty."
                print(f"Warning: {last_error}")

            except Exception as e:
                if 'api_call_start' in locals():
                    api_time_only_sec += time.time() - api_call_start
                last_error = f"Error in Anthropic API call (attempt {attempt+1}/{self.config.max_retries}): {str(e)}"
                print(last_error)
                if attempt == self.config.max_retries - 1:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(None)
                    return f"Anthropic API call failed after {self.config.max_retries} attempts. Last error: {last_error}"
                random_sleep = random.uniform(2 ** attempt, 2 ** (attempt + 1))
                print(f"Waiting {random_sleep} seconds before retry...")
                time.sleep(random_sleep)
            finally:
                time.sleep(1.1)

        self._set_last_call_timing(api_time_only_sec, attempt_count)
        self._set_last_call_usage(None)
        return f"Anthropic API call failed after {self.config.max_retries} attempts. Last error: {last_error}"

    @staticmethod
    def _normalize_tool_arguments(arguments: Any) -> Dict[str, Any]:
        """Normalize serialized tool-call arguments into a dict."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            return json.loads(arguments)
        raise ValueError("Tool arguments must be a dict or JSON string.")

    @staticmethod
    def _parse_tool_result_content(content: Any) -> str:
        """Anthropic tool_result accepts text content; normalize to string."""
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _convert_openai_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style function tools to Anthropic tools."""
        anthropic_tools = []
        for tool in tools:
            function_def = tool.get("function", {})
            anthropic_tools.append({
                "name": function_def.get("name"),
                "description": function_def.get("description", ""),
                "input_schema": function_def.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    @staticmethod
    def _normalize_anthropic_tool_choice(
        tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """Map common tool choice values to Anthropic's tool_choice shape."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            normalized = tool_choice.lower()
            if normalized in {"auto", "any", "none"}:
                return {"type": normalized}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function_obj = tool_choice.get("function", {})
                name = function_obj.get("name")
                if name:
                    return {"type": "tool", "name": name}
            choice_type = tool_choice.get("type")
            if isinstance(choice_type, str) and choice_type.lower() in {"auto", "any", "none", "tool"}:
                normalized = {"type": choice_type.lower()}
                if normalized["type"] == "tool" and tool_choice.get("name"):
                    normalized["name"] = tool_choice["name"]
                return normalized
        return None

    def _convert_common_messages_to_anthropic(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert common chat/tool history into Anthropic Messages format."""
        anthropic_messages: List[Dict[str, Any]] = []
        pending_tool_results: List[Dict[str, Any]] = []

        def flush_pending_tool_results() -> None:
            if pending_tool_results:
                anthropic_messages.append({
                    "role": "user",
                    "content": list(pending_tool_results),
                })
                pending_tool_results.clear()

        for message in messages:
            role = message.get("role")
            if role == "tool":
                raw_content = message.get("content")
                parsed_content = None
                try:
                    parsed_content = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                except json.JSONDecodeError:
                    parsed_content = raw_content
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id"),
                    "content": self._parse_tool_result_content(raw_content),
                }
                if isinstance(parsed_content, dict) and parsed_content.get("status") == "error":
                    tool_result_block["is_error"] = True
                pending_tool_results.append(tool_result_block)
                continue

            flush_pending_tool_results()
            if role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": message.get("content") or ""}],
                })
                continue

            if role == "assistant":
                content_blocks: List[Dict[str, Any]] = []
                text_content = message.get("content")
                if text_content:
                    content_blocks.append({"type": "text", "text": text_content})
                for idx, tool_call in enumerate(message.get("tool_calls") or [], start=1):
                    function_obj = tool_call.get("function", {})
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tool_call.get("id") or f"toolu_{idx}",
                        "name": function_obj.get("name"),
                        "input": self._normalize_tool_arguments(function_obj.get("arguments")),
                    })
                if content_blocks:
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })

        flush_pending_tool_results()
        return anthropic_messages

    def call_with_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call Anthropic Messages API with structured messages/tools."""
        self._clear_last_call_metadata()
        api_time_only_sec = 0.0
        attempt_count = 0
        last_error = None

        if not self.api_key:
            self._set_last_call_timing(api_time_only_sec, attempt_count)
            self._set_last_call_usage(None)
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
                "finish_reason": "error",
                "error": "Anthropic API key is missing.",
            }

        for attempt in range(self.config.max_retries):
            try:
                attempt_count += 1
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload: Dict[str, Any] = {
                    "model": self.config.model_name,
                    "max_tokens": getattr(self.config, "anthropic_max_tokens", 512),
                    "temperature": self.config.temperature,
                    "messages": self._convert_common_messages_to_anthropic(messages),
                }
                if tools:
                    payload["tools"] = self._convert_openai_tools_to_anthropic(tools)
                normalized_tool_choice = self._normalize_anthropic_tool_choice(tool_choice)
                if normalized_tool_choice is not None:
                    payload["tool_choice"] = normalized_tool_choice

                api_call_start = time.time()
                response = requests.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=self.config.openai_compatible_timeout_sec,
                )
                api_time_only_sec += time.time() - api_call_start

                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

                body = response.json()
                content = body.get("content", [])
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                tool_calls = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                            },
                        })

                response_text = "".join(text_parts).strip()
                if response_text or tool_calls:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    usage = body.get("usage") or {}
                    self._set_last_call_usage(self._build_usage_obj(usage))
                    finish_reason = body.get("stop_reason")
                    if tool_calls and finish_reason is None:
                        finish_reason = "tool_use"
                    return {
                        "role": body.get("role", "assistant"),
                        "content": response_text or None,
                        "tool_calls": tool_calls,
                        "finish_reason": finish_reason,
                    }

                last_error = "Anthropic structured response was empty."
                print(f"Warning: {last_error}")
            except Exception as e:
                if 'api_call_start' in locals():
                    api_time_only_sec += time.time() - api_call_start
                last_error = f"Error in Anthropic structured API call (attempt {attempt+1}/{self.config.max_retries}): {str(e)}"
                print(last_error)
                if attempt == self.config.max_retries - 1:
                    self._set_last_call_timing(api_time_only_sec, attempt_count)
                    self._set_last_call_usage(None)
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                        "finish_reason": "error",
                        "error": last_error,
                    }
                random_sleep = random.uniform(2 ** attempt, 2 ** (attempt + 1))
                print(f"Waiting {random_sleep} seconds before retry...")
                time.sleep(random_sleep)
            finally:
                time.sleep(1.1)

        self._set_last_call_timing(api_time_only_sec, attempt_count)
        self._set_last_call_usage(None)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
            "finish_reason": "error",
            "error": last_error,
        }

def get_api_client(provider: str, config: APIConfig, **kwargs) -> APIClient:
    """Factory function to get the appropriate API client based on provider"""
    if provider.lower() == "gemini":
        api_key = kwargs.get('api_key')
        api_keys = kwargs.get('api_keys')  # Support multiple keys
        return GeminiClient(config, api_key, api_keys)
    # elif provider.lower() == "gpt":
    #     return GPTClient(config)
    elif provider.lower() == "azure-openai" or provider.lower() == "azure":
        # Extract Azure-specific parameters from kwargs
        endpoint = kwargs.get('endpoint')
        deployment = kwargs.get('deployment')
        managed_identity_client_id = kwargs.get('managed_identity_client_id')
        return AzureOpenAIClient(config, endpoint, deployment, managed_identity_client_id)
    elif provider.lower() in ("openai-compatible", "openai_compatible"):
        api_key = kwargs.get('api_key')
        base_url = kwargs.get('base_url')
        return OpenAICompatibleClient(config, api_key=api_key, base_url=base_url)
    elif provider.lower() in ("anthropic", "claude"):
        api_key = kwargs.get('api_key')
        base_url = kwargs.get('base_url')
        return AnthropicClient(config, api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Unsupported API provider: {provider}") 
