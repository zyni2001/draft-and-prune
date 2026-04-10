#!/usr/bin/env python3
import os
import yaml
from datetime import datetime
from typing import Optional, Dict, Any

class ReasonerConfig:
    """Configuration for the reasoning system"""
    
    def __init__(self, 
                 dataset: str = None,
                 test_file: str = None,
                 reasoning_method: str = None,
                 api_key: str = None,
                 provider: str = None,
                 model_providers: Dict[str, str] = None,
                 model: str = "gemini-2.5-flash-preview-04-17",  # Backward compatibility
                 plan_model: str = None,  # New: Model for plan generation
                 code_model: str = None,  # New: Model for code generation
                 code_temp: float = 0.6, 
                 max_repairs: int = 3,
                 max_retries: int = 10,
                 num_processes: int = 1,
                 test_delay: int = 5,
                 prompt_path: str = None,
                 shots: str = 'zero',
                 desired_indices: list = None,
                 force_tool_name: str = None,
                 # Azure OpenAI specific parameters
                 azure_endpoint: str = None,
                 azure_deployment: str = None,
                 azure_managed_identity_client_id: str = None,
                 gemini_api_key: str = None,
                 # Gemini multiple API keys
                 gemini_api_keys: list = None,
                 gemini_thinking_budget: int = 0,
                 gemini_thinking_level: str = None,
                 # OpenAI-compatible provider settings
                 openai_compatible_base_url: str = None,
                 openai_compatible_api_key: str = None,
                 openai_compatible_timeout_sec: int = 60,
                 openai_compatible_max_tokens: int = None,
                 openai_compatible_default_extra_body: Dict[str, Any] = None,
                 openai_compatible_extra_body: Dict[str, Dict[str, Any]] = None,
                 # Anthropic provider settings
                 anthropic_api_key: str = None,
                 anthropic_base_url: str = None,
                 anthropic_max_tokens: int = 512,
                 # Plan generation parameters
                 num_paths: int = 3,
                 plan_temp: float = 1.0,
                 # Result storage
                 results_root: str = "./results"):
        """
        Initialize configuration either from parameters or will be loaded from YAML
        
        Args:
            reasoning_method: The reasoning method to use
            api_key: API key for the model service
            dataset: Dataset name to use
            test_file: Test file to use
            model: Primary model name
            code_temp: code_temp for model generation
            max_repairs: Maximum number of repair attempts
            max_retries: Maximum number of LLM requests
            num_processes: Number of parallel processes to use
            test_delay: Delay between test cases in seconds
            prompt_path: Path to the prompt file
            shots: Number of shots for few-shot learning
            desired_indices: Desired indices to use
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure OpenAI deployment
            azure_managed_identity_client_id: Azure managed identity client ID
            num_paths: Number of different plans to generate
            plan_temp: code_temp for plan generation
            results_root: Root directory for experiment outputs
        """
        self.dataset = dataset
        self.test_file = test_file
        self.reasoning_method = reasoning_method
        self.api_key = api_key
        self.provider = provider
        self.model_providers = model_providers
        self.model = model  # Backward compatibility
        self.plan_model = plan_model
        self.code_model = code_model
        self.code_temp = code_temp
        self.max_repairs = max_repairs
        self.max_retries = max_retries
        self.num_processes = num_processes
        self.test_delay = test_delay
        self.prompt_path = prompt_path
        self.shots = shots
        self.desired_indices = desired_indices
        self.force_tool_name = force_tool_name
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_managed_identity_client_id = azure_managed_identity_client_id
        self.gemini_api_key = gemini_api_key
        self.gemini_api_keys = gemini_api_keys
        self.gemini_thinking_budget = gemini_thinking_budget
        self.gemini_thinking_level = gemini_thinking_level
        self.openai_compatible_base_url = openai_compatible_base_url
        self.openai_compatible_api_key = openai_compatible_api_key
        self.openai_compatible_timeout_sec = openai_compatible_timeout_sec
        self.openai_compatible_max_tokens = openai_compatible_max_tokens
        self.openai_compatible_default_extra_body = openai_compatible_default_extra_body
        self.openai_compatible_extra_body = openai_compatible_extra_body
        self.anthropic_api_key = anthropic_api_key
        self.anthropic_base_url = anthropic_base_url
        self.anthropic_max_tokens = anthropic_max_tokens
        self.num_paths = num_paths
        self.plan_temp = plan_temp
        self.results_root = results_root
    
    @classmethod
    def _resolve_env_value(cls, value: Any) -> Any:
        """Resolve ${ENV_VAR} style placeholders in string values."""
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ReasonerConfig':
        """
        Load configuration from a YAML file
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            ReasonerConfig instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            KeyError: If required configuration keys are missing
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")
        
        if config_data is None:
            raise ValueError(f"Empty or invalid YAML file: {yaml_path}")
        
        # Create instance with loaded data
        instance = cls()
        instance.dataset = config_data['dataset']
        instance.test_file = config_data['test_file']
        instance.reasoning_method = config_data['reasoning_method']
        instance.api_key = cls._resolve_env_value(config_data['api_key'])
        instance.provider = config_data.get('provider')
        instance.model_providers = config_data.get('model_providers')
        
        # Handle new model field names with backward compatibility
        if 'plan_model' in config_data:
            instance.plan_model = config_data['plan_model']
            instance.model = config_data['plan_model']  # For backward compatibility
        else:
            instance.model = config_data.get('model', 'gemini-2.5-flash')
            instance.plan_model = instance.model
            
        if 'code_model' in config_data:
            instance.code_model = config_data['code_model']
        else:
            instance.code_model = instance.plan_model  # Default to same as plan model
        instance.code_temp = config_data['code_temp']
        instance.max_repairs = config_data['max_repairs']
        if config_data.get('max_retries'):
            instance.max_retries = config_data['max_retries']
        else:
            instance.max_retries = 10   # Set Default Value
        instance.num_processes = config_data.get('num_processes', 1)  # Default to 1 if not specified
        instance.test_delay = config_data['test_delay']
        instance.prompt_path = config_data['prompt_path']
        instance.shots = config_data['shots']
        instance.desired_indices = config_data['desired_indices']
        instance.force_tool_name = config_data.get('force_tool_name')
        # Azure OpenAI fields are optional for backward compatibility
        instance.azure_endpoint = config_data.get('azure_endpoint')
        instance.azure_deployment = config_data.get('azure_deployment')
        instance.azure_managed_identity_client_id = config_data.get('azure_managed_identity_client_id')
        # Gemini multiple API keys are optional
        gemini_api_key = config_data.get('gemini_api_key') or config_data.get('api_key')
        instance.gemini_api_key = cls._resolve_env_value(gemini_api_key)
        raw_gemini_keys = config_data.get('gemini_api_keys')
        if isinstance(raw_gemini_keys, list):
            instance.gemini_api_keys = [cls._resolve_env_value(k) for k in raw_gemini_keys]
        else:
            instance.gemini_api_keys = raw_gemini_keys
        instance.gemini_thinking_budget = config_data.get('gemini_thinking_budget', 0)
        instance.gemini_thinking_level = config_data.get('gemini_thinking_level')
        instance.openai_compatible_base_url = config_data.get('openai_compatible_base_url')
        openai_compatible_api_key = config_data.get('openai_compatible_api_key')
        instance.openai_compatible_api_key = cls._resolve_env_value(openai_compatible_api_key)
        instance.openai_compatible_timeout_sec = config_data.get('openai_compatible_timeout_sec', 60)
        instance.openai_compatible_max_tokens = config_data.get('openai_compatible_max_tokens')
        instance.openai_compatible_default_extra_body = config_data.get('openai_compatible_default_extra_body')
        instance.openai_compatible_extra_body = config_data.get('openai_compatible_extra_body')
        anthropic_api_key = config_data.get('anthropic_api_key')
        instance.anthropic_api_key = cls._resolve_env_value(anthropic_api_key)
        instance.anthropic_base_url = config_data.get('anthropic_base_url')
        instance.anthropic_max_tokens = config_data.get('anthropic_max_tokens', 512)
        # Plan generation parameters with defaults
        instance.num_paths = config_data.get('num_paths', 3)
        instance.plan_temp = config_data.get('plan_temp', 1.0)
        # Results root with default
        instance.results_root = config_data.get('results_root', "./results")
        instance._validate_config()
        return instance
    
    def _validate_config(self) -> None:
        """Validate configuration values and types"""
        allowed_providers = {"gemini", "azure-openai", "openai-compatible", "azure", "openai_compatible", "anthropic", "claude"}
        allowed_reasoning_methods = {"cot", "one-step", "two-step", "three-step", "adaptive-agent"}
        if not isinstance(self.reasoning_method, str):
            raise TypeError("reasoning_method must be a string")
        if self.reasoning_method not in allowed_reasoning_methods:
            raise ValueError("reasoning_method must be one of: cot, one-step, two-step, three-step, adaptive-agent")
        if not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        if self.provider is not None and not isinstance(self.provider, str):
            raise TypeError("provider must be a string or None")
        if isinstance(self.provider, str) and self.provider.lower() not in allowed_providers:
            raise ValueError("provider must be one of: gemini, azure-openai, openai-compatible, anthropic")
        if self.model_providers is not None:
            if not isinstance(self.model_providers, dict):
                raise TypeError("model_providers must be a dictionary or None")
            for model_name, provider_name in self.model_providers.items():
                if not isinstance(model_name, str) or not isinstance(provider_name, str):
                    raise TypeError("model_providers keys and values must be strings")
                if provider_name.lower() not in allowed_providers:
                    raise ValueError(
                        f"model_providers['{model_name}'] must be one of: gemini, azure-openai, openai-compatible, anthropic"
                    )
        if not isinstance(self.dataset, str):
            raise TypeError("dataset must be a string")
        if not isinstance(self.model, str):
            raise TypeError("model must be a string")
        if not isinstance(self.code_temp, (int, float)) or self.code_temp < 0:
            raise ValueError("code_temp must be a non-negative number")
        if not isinstance(self.max_repairs, int) or self.max_repairs < 0:
            raise ValueError("max_repairs must be a non-negative integer")
        if not isinstance(self.max_retries, int) or self.max_retries <= 0:
            raise ValueError("max_retries must be a positive integer")
        if not isinstance(self.num_processes, int) or self.num_processes <= 0:
            raise ValueError("num_processes must be a positive integer")
        if not isinstance(self.test_delay, int) or self.test_delay < 0:
            raise ValueError("test_delay must be a non-negative integer")
        if not isinstance(self.shots, str) or self.shots not in ['zero', 'one', 'two', 'three', 'six', 'nine']:
            raise ValueError("shots must be a string in ['zero', 'one', 'two', 'three', 'six', 'nine']")
        if self.desired_indices is not None and not isinstance(self.desired_indices, list):
            raise TypeError("desired_indices must be a list or None")
        if self.force_tool_name is not None and not isinstance(self.force_tool_name, str):
            raise TypeError("force_tool_name must be a string or None")
        if self.azure_endpoint is not None and not isinstance(self.azure_endpoint, str):
            raise TypeError("azure_endpoint must be a string or None")
        if self.azure_deployment is not None and not isinstance(self.azure_deployment, str):
            raise TypeError("azure_deployment must be a string or None")
        if self.azure_managed_identity_client_id is not None and not isinstance(self.azure_managed_identity_client_id, str):
            raise TypeError("azure_managed_identity_client_id must be a string or None")
        if self.gemini_api_key is not None and not isinstance(self.gemini_api_key, str):
            raise TypeError("gemini_api_key must be a string or None")
        if self.gemini_thinking_budget is not None:
            if not isinstance(self.gemini_thinking_budget, int):
                raise TypeError("gemini_thinking_budget must be an integer or None")
            if self.gemini_thinking_budget < 0:
                raise ValueError("gemini_thinking_budget must be >= 0")
        if self.gemini_thinking_level is not None:
            if not isinstance(self.gemini_thinking_level, str):
                raise TypeError("gemini_thinking_level must be a string or None")
            if self.gemini_thinking_level.lower() not in {"minimal", "low", "medium", "high"}:
                raise ValueError("gemini_thinking_level must be one of: minimal, low, medium, high")
        if self.gemini_thinking_level is not None and self.gemini_thinking_budget not in (None, 0):
            raise ValueError("Set only one of gemini_thinking_level or gemini_thinking_budget (non-zero)")
        if self.openai_compatible_base_url is not None and not isinstance(self.openai_compatible_base_url, str):
            raise TypeError("openai_compatible_base_url must be a string or None")
        if self.openai_compatible_api_key is not None and not isinstance(self.openai_compatible_api_key, str):
            raise TypeError("openai_compatible_api_key must be a string or None")
        if self.openai_compatible_timeout_sec is not None:
            if not isinstance(self.openai_compatible_timeout_sec, int):
                raise TypeError("openai_compatible_timeout_sec must be an integer or None")
            if self.openai_compatible_timeout_sec <= 0:
                raise ValueError("openai_compatible_timeout_sec must be > 0")
        if self.openai_compatible_max_tokens is not None:
            if not isinstance(self.openai_compatible_max_tokens, int):
                raise TypeError("openai_compatible_max_tokens must be an integer or None")
            if self.openai_compatible_max_tokens <= 0:
                raise ValueError("openai_compatible_max_tokens must be > 0")
        if self.openai_compatible_default_extra_body is not None:
            if not isinstance(self.openai_compatible_default_extra_body, dict):
                raise TypeError("openai_compatible_default_extra_body must be a dictionary or None")
        if self.openai_compatible_extra_body is not None:
            if not isinstance(self.openai_compatible_extra_body, dict):
                raise TypeError("openai_compatible_extra_body must be a dictionary or None")
            for model_name, extra_body in self.openai_compatible_extra_body.items():
                if not isinstance(model_name, str):
                    raise TypeError("openai_compatible_extra_body keys must be model name strings")
                if not isinstance(extra_body, dict):
                    raise TypeError("openai_compatible_extra_body values must be dictionaries")
        if self.anthropic_api_key is not None and not isinstance(self.anthropic_api_key, str):
            raise TypeError("anthropic_api_key must be a string or None")
        if self.anthropic_base_url is not None and not isinstance(self.anthropic_base_url, str):
            raise TypeError("anthropic_base_url must be a string or None")
        if self.anthropic_max_tokens is not None:
            if not isinstance(self.anthropic_max_tokens, int):
                raise TypeError("anthropic_max_tokens must be an integer or None")
            if self.anthropic_max_tokens <= 0:
                raise ValueError("anthropic_max_tokens must be > 0")
        if not isinstance(self.num_paths, int) or self.num_paths <= 0:
            raise ValueError("num_paths must be a positive integer")
        if not isinstance(self.plan_temp, (int, float)) or self.plan_temp < 0:
            raise ValueError("plan_temp must be a non-negative number")
        if not isinstance(self.results_root, str) or not self.results_root.strip():
            raise ValueError("results_root must be a non-empty string")
    
    def save_to_yaml(self, yaml_path: str) -> None:
        """
        Save current configuration to a YAML file
        
        Args:
            yaml_path: Path where to save the YAML configuration file
        """
        # Create ordered dictionary with both new and legacy field names
        ordered_config = {
            'dataset': self.dataset,
            'test_file': self.test_file,
            'reasoning_method': self.reasoning_method,
            'api_key': self.api_key,
            'provider': getattr(self, 'provider', None),
            'model_providers': getattr(self, 'model_providers', None),
            # New model field names (preferred)
            'plan_model': getattr(self, 'plan_model', self.model),
            'code_model': getattr(self, 'code_model', self.model),
            # Legacy field names (for backward compatibility)
            'model': self.model,
            'code_temp': self.code_temp,
            'max_repairs': self.max_repairs,
            'max_retries': self.max_retries,
            'num_processes': self.num_processes,
            'test_delay': self.test_delay,
            'prompt_path': self.prompt_path,
            'shots': self.shots,
            'desired_indices': self.desired_indices,
            'force_tool_name': getattr(self, 'force_tool_name', None),
            'azure_endpoint': self.azure_endpoint,
            'azure_deployment': self.azure_deployment,
            'azure_managed_identity_client_id': self.azure_managed_identity_client_id,
            'gemini_api_key': getattr(self, 'gemini_api_key', None),
            # Multiple Gemini API keys for batch distribution
            'gemini_api_keys': getattr(self, 'gemini_api_keys', None),
            'gemini_thinking_budget': getattr(self, 'gemini_thinking_budget', 0),
            'gemini_thinking_level': getattr(self, 'gemini_thinking_level', None),
            'openai_compatible_base_url': getattr(self, 'openai_compatible_base_url', None),
            'openai_compatible_api_key': getattr(self, 'openai_compatible_api_key', None),
            'openai_compatible_timeout_sec': getattr(self, 'openai_compatible_timeout_sec', 60),
            'openai_compatible_max_tokens': getattr(self, 'openai_compatible_max_tokens', None),
            'openai_compatible_default_extra_body': getattr(self, 'openai_compatible_default_extra_body', None),
            'openai_compatible_extra_body': getattr(self, 'openai_compatible_extra_body', None),
            'anthropic_api_key': getattr(self, 'anthropic_api_key', None),
            'anthropic_base_url': getattr(self, 'anthropic_base_url', None),
            'anthropic_max_tokens': getattr(self, 'anthropic_max_tokens', 512),
            # Plan generation parameters
            'num_paths': getattr(self, 'num_paths', 3),
            'plan_temp': getattr(self, 'plan_temp', 1.0),
            # Result storage
            'results_root': getattr(self, 'results_root', "./results")
        }
        
        try:
            with open(yaml_path, 'w', encoding='utf-8') as file:
                yaml.dump(ordered_config, file, default_flow_style=False, indent=2, sort_keys=False)
        except Exception as e:
            raise IOError(f"Error saving configuration to {yaml_path}: {e}")
    
    def __str__(self) -> str:
        """String representation of the configuration"""
        return f"ReasonerConfig({self.__dict__})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the configuration"""
        return self.__str__()
