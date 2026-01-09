#!/usr/bin/env python3
"""
Test to verify the Google GenAI transformation logic for generateContent parameters
"""
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest

from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_map_generate_content_optional_params_response_json_schema_camelcase():
    """Test that responseJsonSchema (camelCase) is passed through correctly"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "responseJsonSchema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-3-flash-preview"
    )
    
    # responseJsonSchema should be in the result (camelCase format for Google GenAI API)
    assert "responseJsonSchema" in result
    assert result["responseJsonSchema"] == generate_content_config_dict["responseJsonSchema"]
    assert "temperature" in result
    assert result["temperature"] == 1.0


def test_map_generate_content_optional_params_response_schema_snakecase():
    """Test that response_schema (snake_case) is converted to responseJsonSchema (camelCase)"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "response_json_schema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-3-flash-preview"
    )
    
    # response_schema should be converted to responseJsonSchema (camelCase)
    assert "responseJsonSchema" in result
    assert result["responseJsonSchema"] == generate_content_config_dict["response_json_schema"]
    assert "temperature" in result


def test_map_generate_content_optional_params_thinking_config_camelcase():
    """Test that thinkingConfig (camelCase) is passed through correctly"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "thinkingConfig": {
            "thinkingLevel": "minimal",
            "includeThoughts": True
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-3-flash-preview"
    )
    
    # thinkingConfig should be in the result (camelCase format for Google GenAI API)
    assert "thinkingConfig" in result
    assert result["thinkingConfig"]["thinkingLevel"] == "minimal"
    assert result["thinkingConfig"]["includeThoughts"] is True
    assert "temperature" in result


def test_map_generate_content_optional_params_thinking_config_snakecase():
    """Test that thinking_config (snake_case) is converted to thinkingConfig (camelCase)"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "thinking_config": {
            "thinkingLevel": "medium",
            "includeThoughts": True
        },
        "temperature": 1.0
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-3-flash-preview"
    )
    
    # thinking_config should be converted to thinkingConfig (camelCase)
    assert "thinkingConfig" in result
    assert result["thinkingConfig"]["thinkingLevel"] == "medium"
    assert result["thinkingConfig"]["includeThoughts"] is True
    assert "thinking_config" not in result  # Should not be in snake_case format
    assert "temperature" in result


def test_map_generate_content_optional_params_mixed_formats():
    """Test that both camelCase and snake_case parameters work together"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "responseJsonSchema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        },
        "thinking_config": {
            "thinkingLevel": "low",
            "includeThoughts": True
        },
        "temperature": 1.0,
        "max_output_tokens": 100
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-3-flash-preview"
    )
    
    # All parameters should be converted to camelCase
    assert "responseJsonSchema" in result
    assert "thinkingConfig" in result
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert "temperature" in result
    assert "maxOutputTokens" in result  # This one stays as-is if it's in supported list


def test_map_generate_content_optional_params_response_mime_type():
    """Test that responseMimeType is handled correctly"""
    config = GoogleGenAIConfig()
    
    generate_content_config_dict = {
        "responseMimeType": "application/json",
        "responseJsonSchema": {
            "type": "object",
            "properties": {
                "recipe_name": {"type": "string"}
            }
        }
    }
    
    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-3-flash-preview"
    )
    
    # responseMimeType should be passed through (it's already camelCase)
    assert "responseMimeType" in result or "response_mime_type" in result
    assert "responseJsonSchema" in result


def test_responses_api_reasoning_dict_format():
    """Test that reasoning parameter with dict format is mapped to reasoning_effort"""
    from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
    
    responses_api_request: ResponsesAPIOptionalRequestParams = {
        "reasoning": {"effort": "high"},
        "temperature": 1.0,
    }
    
    result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="gemini/2.5-pro",
        input="Hello, what is the capital of France?",
        responses_api_request=responses_api_request,
    )
    
    # reasoning_effort should be extracted from reasoning dict
    assert "reasoning_effort" in result
    assert result["reasoning_effort"] == "high"


def test_responses_api_reasoning_string_format():
    """Test that reasoning parameter with string format is mapped to reasoning_effort"""
    from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
    
    responses_api_request: ResponsesAPIOptionalRequestParams = {
        "reasoning": "medium",  # Could be a string directly
        "temperature": 1.0,
    }
    
    result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="gemini/2.5-pro",
        input="Hello, what is the capital of France?",
        responses_api_request=responses_api_request,
    )
    
    # reasoning_effort should be extracted from reasoning string
    assert "reasoning_effort" in result
    assert result["reasoning_effort"] == "medium"


def test_responses_api_reasoning_low_effort():
    """Test that low reasoning effort is correctly mapped"""
    from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
    
    responses_api_request: ResponsesAPIOptionalRequestParams = {
        "reasoning": {"effort": "low"},
    }
    
    result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="gemini/2.5-pro",
        input="Test",
        responses_api_request=responses_api_request,
    )
    
    assert "reasoning_effort" in result
    assert result["reasoning_effort"] == "low"


def test_responses_api_no_reasoning():
    """Test that no reasoning_effort is included when reasoning is not provided"""
    from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
    
    responses_api_request: ResponsesAPIOptionalRequestParams = {
        "temperature": 1.0,
    }
    
    result = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="gemini/2.5-pro",
        input="Test",
        responses_api_request=responses_api_request,
    )
    
    # reasoning_effort should not be in result if not provided (filtered out as None)
    assert "reasoning_effort" not in result or result.get("reasoning_effort") is None


def test_transform_generate_content_request_with_system_instruction():
    """Test that systemInstruction parameter is properly included in the request"""
    config = GoogleGenAIConfig()
    
    system_instruction = {
        "parts": [{"text": "You are a helpful assistant"}]
    }
    
    contents = [
        {
            "role": "user",
            "parts": [{"text": "Hello"}]
        }
    ]
    
    generate_content_config_dict = {
        "temperature": 1.0,
        "maxOutputTokens": 100
    }
    
    # Call transform_generate_content_request
    result = config.transform_generate_content_request(
        model="gemini-3-flash-preview",
        contents=contents,
        tools=None,
        generate_content_config_dict=generate_content_config_dict,
        system_instruction=system_instruction,
    )
    
    # Verify that systemInstruction is in the request
    assert "systemInstruction" in result, "systemInstruction should be in request body"
    assert result["systemInstruction"] == system_instruction, "systemInstruction should match input"
    assert result["model"] == "gemini-3-flash-preview"
    assert result["contents"] == contents


def test_transform_generate_content_request_without_system_instruction():
    """Test that request works correctly without systemInstruction"""
    config = GoogleGenAIConfig()
    
    contents = [
        {
            "role": "user",
            "parts": [{"text": "Hello"}]
        }
    ]
    
    generate_content_config_dict = {
        "temperature": 1.0
    }
    
    # Call transform_generate_content_request without system_instruction
    result = config.transform_generate_content_request(
        model="gemini-3-flash-preview",
        contents=contents,
        tools=None,
        generate_content_config_dict=generate_content_config_dict,
        system_instruction=None,
    )
    
    # Verify that systemInstruction is NOT in the request when not provided
    assert "systemInstruction" not in result, "systemInstruction should not be in request when None"
    assert result["model"] == "gemini-3-flash-preview"
    assert result["contents"] == contents


def test_transform_generate_content_request_system_instruction_with_tools():
    """Test that systemInstruction works correctly alongside tools"""
    config = GoogleGenAIConfig()
    
    system_instruction = {
        "parts": [{"text": "You are a helpful assistant that uses tools"}]
    }
    
    contents = [
        {
            "role": "user",
            "parts": [{"text": "What's the weather?"}]
        }
    ]
    
    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            ]
        }
    ]
    
    generate_content_config_dict = {
        "temperature": 0.7
    }
    
    # Call transform_generate_content_request with both system_instruction and tools
    result = config.transform_generate_content_request(
        model="gemini-3-flash-preview",
        contents=contents,
        tools=tools,
        generate_content_config_dict=generate_content_config_dict,
        system_instruction=system_instruction,
    )
    
    # Verify that both systemInstruction and tools are in the request
    assert "systemInstruction" in result, "systemInstruction should be in request body"
    assert result["systemInstruction"] == system_instruction
    assert "tools" in result, "tools should be in request body"
    assert result["tools"] == tools
    assert result["model"] == "gemini-3-flash-preview"


def test_validate_environment_with_dict_api_key():
    """
    Test that validate_environment correctly handles api_key as a dict.
    
    This happens when using custom api_base with Gemini - the auth_header
    is returned as {"x-goog-api-key": "sk-test"} and should be merged into
    headers instead of being set as a header value.
    
    Regression test for: https://github.com/BerriAI/litellm/issues/xxxxx
    """
    config = GoogleGenAIConfig()
    
    # Simulate the case where auth_header is a dict (custom api_base scenario)
    auth_header_dict = {"x-goog-api-key": "sk-test-key-123"}
    
    result = config.validate_environment(
        api_key=auth_header_dict,
        headers=None,
        model="gemini-2.5-pro",
        litellm_params={}
    )
    
    # The dict should be merged into headers, not set as a value
    assert "x-goog-api-key" in result, "x-goog-api-key should be in headers"
    assert result["x-goog-api-key"] == "sk-test-key-123", "API key should be the string value, not a dict"
    assert isinstance(result["x-goog-api-key"], str), "Header value should be a string, not a dict"
    assert "Content-Type" in result, "Content-Type should be in headers"
    assert result["Content-Type"] == "application/json"


def test_validate_environment_with_string_api_key():
    """
    Test that validate_environment correctly handles api_key as a string.
    
    This is the normal case when using standard Gemini API.
    """
    config = GoogleGenAIConfig()
    
    # Normal case: api_key is a string
    api_key_string = "sk-test-key-456"
    
    result = config.validate_environment(
        api_key=api_key_string,
        headers=None,
        model="gemini-2.5-pro",
        litellm_params={}
    )
    
    # The string should be set as the header value
    assert "x-goog-api-key" in result, "x-goog-api-key should be in headers"
    assert result["x-goog-api-key"] == "sk-test-key-456", "API key should match input"
    assert isinstance(result["x-goog-api-key"], str), "Header value should be a string"
    assert "Content-Type" in result, "Content-Type should be in headers"


def test_validate_environment_with_extra_headers():
    """
    Test that validate_environment correctly merges extra headers with dict api_key.
    """
    config = GoogleGenAIConfig()
    
    # Custom api_base scenario with additional headers
    auth_header_dict = {"x-goog-api-key": "sk-test-key-789"}
    extra_headers = {"X-Custom-Header": "custom-value"}
    
    result = config.validate_environment(
        api_key=auth_header_dict,
        headers=extra_headers,
        model="gemini-2.5-pro",
        litellm_params={}
    )
    
    # Both the auth dict and extra headers should be merged
    assert "x-goog-api-key" in result, "x-goog-api-key should be in headers"
    assert result["x-goog-api-key"] == "sk-test-key-789", "API key should be correctly set"
    assert isinstance(result["x-goog-api-key"], str), "Header value should be a string"
    assert "X-Custom-Header" in result, "Extra headers should be merged"
    assert result["X-Custom-Header"] == "custom-value"
    assert "Content-Type" in result


# =============================================================================
# tool_choice transformation tests
# =============================================================================


def test_map_generate_content_optional_params_tool_choice_required():
    """Test that tool_choice='required' is stored for later transformation"""
    config = GoogleGenAIConfig()

    generate_content_config_dict = {
        "tool_choice": "required",
        "temperature": 1.0
    }

    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-2.5-flash"
    )

    assert "_litellm_tool_choice" in result
    assert result["_litellm_tool_choice"] == "required"
    assert "temperature" in result
    assert result["temperature"] == 1.0


def test_map_generate_content_optional_params_tool_choice_auto():
    """Test that tool_choice='auto' is stored for later transformation"""
    config = GoogleGenAIConfig()

    generate_content_config_dict = {
        "tool_choice": "auto",
        "temperature": 0.7
    }

    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-2.5-flash"
    )

    assert "_litellm_tool_choice" in result
    assert result["_litellm_tool_choice"] == "auto"
    assert "temperature" in result


def test_map_generate_content_optional_params_tool_choice_none():
    """Test that tool_choice='none' is stored for later transformation"""
    config = GoogleGenAIConfig()

    generate_content_config_dict = {
        "tool_choice": "none",
        "temperature": 0.5
    }

    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-2.5-flash"
    )

    assert "_litellm_tool_choice" in result
    assert result["_litellm_tool_choice"] == "none"
    assert "temperature" in result


def test_map_generate_content_optional_params_tool_choice_dict():
    """Test that tool_choice with specific function dict is stored for later transformation"""
    config = GoogleGenAIConfig()

    generate_content_config_dict = {
        "tool_choice": {"function": {"name": "get_weather"}},
        "temperature": 1.0
    }

    result = config.map_generate_content_optional_params(
        generate_content_config_dict=generate_content_config_dict,
        model="gemini/gemini-2.5-flash"
    )

    assert "_litellm_tool_choice" in result
    assert result["_litellm_tool_choice"] == {"function": {"name": "get_weather"}}


def test_transform_generate_content_request_tool_choice_required():
    """Test that tool_choice='required' is transformed to toolConfig with mode='ANY'"""
    config = GoogleGenAIConfig()

    contents = [{"role": "user", "parts": [{"text": "What's the weather?"}]}]
    generate_content_config_dict = {
        "_litellm_tool_choice": "required",
        "temperature": 1.0
    }

    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict=generate_content_config_dict,
    )

    assert "toolConfig" in result, "toolConfig should be in request"
    assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    # Verify _litellm_tool_choice is not in generationConfig
    assert "_litellm_tool_choice" not in result.get("generationConfig", {})


def test_transform_generate_content_request_tool_choice_auto():
    """Test that tool_choice='auto' is transformed to toolConfig with mode='AUTO'"""
    config = GoogleGenAIConfig()

    contents = [{"role": "user", "parts": [{"text": "What's the weather?"}]}]
    generate_content_config_dict = {
        "_litellm_tool_choice": "auto",
        "temperature": 1.0
    }

    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict=generate_content_config_dict,
    )

    assert "toolConfig" in result, "toolConfig should be in request"
    assert result["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"


def test_transform_generate_content_request_tool_choice_none():
    """Test that tool_choice='none' is transformed to toolConfig with mode='NONE'"""
    config = GoogleGenAIConfig()

    contents = [{"role": "user", "parts": [{"text": "What's the weather?"}]}]
    generate_content_config_dict = {
        "_litellm_tool_choice": "none",
        "temperature": 1.0
    }

    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict=generate_content_config_dict,
    )

    assert "toolConfig" in result, "toolConfig should be in request"
    assert result["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"


def test_transform_generate_content_request_tool_choice_specific_function():
    """Test that tool_choice with specific function is transformed correctly"""
    config = GoogleGenAIConfig()

    contents = [{"role": "user", "parts": [{"text": "What's the weather?"}]}]
    generate_content_config_dict = {
        "_litellm_tool_choice": {"function": {"name": "get_weather"}},
        "temperature": 1.0
    }

    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict=generate_content_config_dict,
    )

    assert "toolConfig" in result, "toolConfig should be in request"
    assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert result["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] == ["get_weather"]


def test_transform_generate_content_request_tool_choice_with_tools():
    """Test that tool_choice works correctly alongside tools parameter"""
    config = GoogleGenAIConfig()

    contents = [{"role": "user", "parts": [{"text": "What's the weather?"}]}]
    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    }
                }
            ]
        }
    ]
    generate_content_config_dict = {
        "_litellm_tool_choice": "required",
        "temperature": 0.7
    }

    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=tools,
        generate_content_config_dict=generate_content_config_dict,
    )

    # Both tools and toolConfig should be present
    assert "tools" in result, "tools should be in request"
    assert result["tools"] == tools
    assert "toolConfig" in result, "toolConfig should be in request"
    assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


def test_transform_generate_content_request_no_tool_choice():
    """Test that no toolConfig is added when tool_choice is not provided"""
    config = GoogleGenAIConfig()

    contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    generate_content_config_dict = {
        "temperature": 1.0
    }

    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict=generate_content_config_dict,
    )

    assert "toolConfig" not in result, "toolConfig should not be in request when tool_choice not provided"


def test_map_tool_choice_to_tool_config_invalid():
    """Test that invalid tool_choice value returns empty dict"""
    config = GoogleGenAIConfig()

    # Test with invalid string value
    result = config._map_tool_choice_to_tool_config("invalid_value")
    assert result == {}, "Invalid tool_choice should return empty dict"

    # Test with invalid dict (missing function key)
    result = config._map_tool_choice_to_tool_config({"invalid": "value"})
    assert result == {} or result["functionCallingConfig"]["allowedFunctionNames"] == [], \
        "Invalid dict should return empty dict or empty allowedFunctionNames"


def test_transform_generate_content_request_tool_choice_dict_not_mutated():
    """Test that the original generate_content_config_dict is not mutated"""
    config = GoogleGenAIConfig()

    contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    original_config = {
        "_litellm_tool_choice": "required",
        "temperature": 1.0
    }
    # Make a copy to compare later
    config_copy = dict(original_config)

    config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict=original_config,
    )

    # Original dict should not be mutated
    assert original_config == config_copy, "Original config dict should not be mutated"


def test_end_to_end_tool_choice_transformation():
    """Test the full flow: map_generate_content_optional_params -> transform_generate_content_request"""
    config = GoogleGenAIConfig()

    # Step 1: User provides tool_choice in config
    user_config = {
        "tool_choice": "required",
        "temperature": 0.8
    }

    # Step 2: map_generate_content_optional_params processes it
    mapped_config = config.map_generate_content_optional_params(
        generate_content_config_dict=user_config,
        model="gemini/gemini-2.5-flash"
    )

    assert "_litellm_tool_choice" in mapped_config

    # Step 3: transform_generate_content_request builds the final request
    contents = [{"role": "user", "parts": [{"text": "Test"}]}]
    result = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,
        tools=None,
        generate_content_config_dict=mapped_config,
    )

    # Final request should have toolConfig at top level with correct mode
    assert "toolConfig" in result
    assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    # generationConfig should have temperature but not _litellm_tool_choice
    assert "temperature" in result["generationConfig"]
    assert "_litellm_tool_choice" not in result["generationConfig"]


def test_consecutive_tool_calls_with_tool_choice():
    """
    Test that consecutive tool calls work correctly with tool_choice.

    Simulates a conversation flow:
    1. User asks a question
    2. Model makes a tool call (get_weather)
    3. Tool response is added to conversation
    4. Model makes another tool call (get_forecast)

    This ensures tool_choice is correctly applied across multiple turns.
    """
    config = GoogleGenAIConfig()

    # Conversation history with tool call and tool response
    contents = [
        # Initial user message
        {"role": "user", "parts": [{"text": "What's the weather in NYC and the forecast for tomorrow?"}]},
        # Model's first tool call
        {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "NYC"}
                    }
                }
            ]
        },
        # Tool response
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "get_weather",
                        "response": {"temperature": 72, "condition": "sunny"}
                    }
                }
            ]
        },
    ]

    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    }
                },
                {
                    "name": "get_forecast",
                    "description": "Get weather forecast",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "days": {"type": "integer"}
                        }
                    }
                }
            ]
        }
    ]

    # First request - should work with tool_choice required
    generate_content_config_dict_1 = {
        "_litellm_tool_choice": "required",
        "temperature": 0.7
    }

    result_1 = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents[:1],  # Just the first user message
        tools=tools,
        generate_content_config_dict=generate_content_config_dict_1,
    )

    assert "toolConfig" in result_1
    assert result_1["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert "tools" in result_1
    assert len(result_1["contents"]) == 1

    # Second request - with full conversation history including tool call and response
    generate_content_config_dict_2 = {
        "_litellm_tool_choice": "required",
        "temperature": 0.7
    }

    result_2 = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents,  # Full conversation with tool call and response
        tools=tools,
        generate_content_config_dict=generate_content_config_dict_2,
    )

    # Verify the second request is properly formed
    assert "toolConfig" in result_2
    assert result_2["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert "tools" in result_2
    assert len(result_2["contents"]) == 3  # user, model (tool call), user (tool response)

    # Verify the conversation history is preserved correctly
    assert result_2["contents"][0]["role"] == "user"
    assert result_2["contents"][1]["role"] == "model"
    assert "functionCall" in result_2["contents"][1]["parts"][0]
    assert result_2["contents"][2]["role"] == "user"
    assert "functionResponse" in result_2["contents"][2]["parts"][0]

    # Verify generationConfig doesn't have _litellm_tool_choice
    assert "_litellm_tool_choice" not in result_2.get("generationConfig", {})


def test_consecutive_tool_calls_config_dict_reused():
    """
    Test that the same config dict can be reused across multiple consecutive calls.

    This is important because in practice, the same config might be passed to
    multiple transform calls during a conversation.
    """
    config = GoogleGenAIConfig()

    # Shared config dict that will be reused
    shared_config = {
        "_litellm_tool_choice": "required",
        "temperature": 0.8
    }

    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            ]
        }
    ]

    # First call
    contents_1 = [{"role": "user", "parts": [{"text": "Search for Python tutorials"}]}]
    result_1 = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents_1,
        tools=tools,
        generate_content_config_dict=shared_config,
    )

    # Second call with same config dict (simulating consecutive turn)
    contents_2 = [
        {"role": "user", "parts": [{"text": "Search for Python tutorials"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "search", "args": {"query": "Python tutorials"}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "search", "response": {"results": ["tutorial1", "tutorial2"]}}}]},
    ]
    result_2 = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents_2,
        tools=tools,
        generate_content_config_dict=shared_config,  # Reuse same config
    )

    # Third call with same config dict
    contents_3 = contents_2 + [
        {"role": "model", "parts": [{"text": "I found some tutorials. Want me to search for more?"}]},
        {"role": "user", "parts": [{"text": "Yes, search for advanced topics"}]},
    ]
    result_3 = config.transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=contents_3,
        tools=tools,
        generate_content_config_dict=shared_config,  # Reuse same config again
    )

    # All three results should have toolConfig correctly set
    for i, result in enumerate([result_1, result_2, result_3], 1):
        assert "toolConfig" in result, f"Call {i}: toolConfig should be present"
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "ANY", f"Call {i}: mode should be ANY"
        assert "_litellm_tool_choice" not in result.get("generationConfig", {}), f"Call {i}: marker should not leak"

    # Verify the shared config was not mutated
    assert "_litellm_tool_choice" in shared_config, "Original config should still have marker"
    assert shared_config["_litellm_tool_choice"] == "required"
