#!/usr/bin/env python3

import os
import sys
import re
import json
import time
import uuid
import shutil
import openai
import tempfile
import traceback
import subprocess
import multiprocessing as mp

from datetime import datetime
from contextlib import redirect_stdout
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union, Literal, Callable
# from pyke import knowledge_engine

from config import ReasonerConfig
from data_loaders import DataLoader
from answer_extractors import AnswerExtractor
from call_api import APIConfig, get_api_client, APIClient


def _parallel_worker(args: Tuple[Any, Dict, Any, int]) -> None:
    """
    Executes the test task in a single subprocess and redirects all standard output to the specified file
    """
    test_runner_instance, test_case, mp_lock, sample_index = args
    
    problem_name = test_case['id_string'] if 'id_string' in test_case else test_case['id']
    unique_id = str(uuid.uuid4())
    log_filepath = os.path.join(test_runner_instance.log_folder, f"{problem_name}-{unique_id}.txt")
    print(f"Start working on {problem_name}...")

    try:
        # Open a log file and redirect stdout to it
        with open(log_filepath, 'w', encoding='utf-8') as log_file:
            with redirect_stdout(log_file):
                # Start recording information in the log
                print(f"--- Log for Task ID: {problem_name} ---")
                print(f"Process ID: {os.getpid()}")
                start_process_time = time.time()
                print(f"Start Time: {datetime.fromtimestamp(start_process_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 30 + "\n")

                # Note: We use multiple API keys for Gemini to avoid rate limiting. 
                # Assign API key based on sample index for batch-based key distribution
                if hasattr(test_runner_instance.config, 'gemini_api_keys') and test_runner_instance.config.gemini_api_keys:
                    num_keys = len(test_runner_instance.config.gemini_api_keys)
                    assigned_key_index = sample_index % num_keys
                    assigned_key = test_runner_instance.config.gemini_api_keys[assigned_key_index]
                    print(f"Sample {sample_index}: Assigned API Key #{assigned_key_index + 1} ({assigned_key[:20]}...)")
                    
                    # Update both main API client and fix API client with the assigned key
                    if hasattr(test_runner_instance.api_client, 'set_api_key'):
                        test_runner_instance.api_client.set_api_key(assigned_key)
                        print(f"Main API client updated with key #{assigned_key_index + 1}")
                    elif hasattr(test_runner_instance.api_client, 'api_key'):
                        test_runner_instance.api_client.api_key = assigned_key
                        print(f"Main API client updated with key #{assigned_key_index + 1}")

                    # Also update fix API client if it exists
                    if hasattr(test_runner_instance, 'code_api_client') and hasattr(test_runner_instance.code_api_client, 'set_api_key'):
                        test_runner_instance.code_api_client.set_api_key(assigned_key)
                        print(f"Fix API client also updated with key #{assigned_key_index + 1}")
                    elif hasattr(test_runner_instance, 'code_api_client') and hasattr(test_runner_instance.code_api_client, 'api_key'):
                        test_runner_instance.code_api_client.api_key = assigned_key
                        print(f"Fix API client also updated with key #{assigned_key_index + 1}")

            # Call the instance's reason method and record per-instance inference time
            case_start_time = time.time()
            reasoning_result = test_runner_instance.reason(test_case, mp_lock)
            case_time = time.time() - case_start_time
            
            test_runner_instance._process_results(test_case, reasoning_result, case_time, unique_id, None)

            print(f"\n" + "-" * 30)
            print(f"Task finished.")

    except Exception as e:
        # If an error occurs during execution, the error message will also be recorded
        with open(log_filepath, 'a', encoding='utf-8') as log_file:
            log_file.write("\n\n****** AN ERROR OCCURRED ******\n")
            log_file.write(traceback.format_exc())
            
    finally:
        print(f"End working on {problem_name}")


class DatasetConfig:
    """Configuration for dataset-specific operations"""
    
    DATASET_SOLVER_MAP = {
        'ar-lsat': 'z3',
        'proofwriter': 'pyke', 
        'folio': 'prover9',
        'proverqa': 'z3',
        'prontoqa': 'pyke',
        'logicaldeduction': 'pythonconstraint'
    }
    
    @classmethod
    def get_solver(cls, dataset: str) -> str:
        """Get the solver name for a dataset"""
        return cls.DATASET_SOLVER_MAP.get(dataset.lower())
    

class PromptHandler:
    """Handles prompt generation for different datasets and reasoning methods"""
    
    def __init__(self, prompt_path: str, dataset: str):
        self.prompt_path = prompt_path
        self.dataset = dataset.lower()
    
    def _load_template(self, template_name: str) -> str:
        """Load a prompt template from file"""
        template_path = os.path.join(self.prompt_path, template_name)
        with open(template_path, "r") as file:
            return file.read()
    
    def _format_prompt(self, template: str, test_case: Dict, **kwargs) -> str:
        """Format a prompt template with test case data"""
        prompt = template
        
        # Common replacements for all datasets
        if 'context' in test_case:
            prompt = prompt.replace("{context}", test_case["context"])
        if 'question' in test_case:
            prompt = prompt.replace("{question}", test_case["question"])
        
        # Dataset-specific replacements
        if self.dataset == "ar-lsat" and 'answers' in test_case:
            prompt = prompt.replace("{answers}", str(test_case["answers"]))
        elif self.dataset in ["proofwriter", "folio", "proverqa", "prontoqa"] and 'options' in test_case:
            prompt = prompt.replace("{options}", str(test_case["options"]))
        elif self.dataset == 'logicaldeduction':
            if 'options' in test_case:
                prompt = prompt.replace("{options}", str(test_case["options"]))
        
        # Additional replacements from kwargs: plan, code, syntax_error
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
        
        return prompt
    
    def get_plan_prompt(self, test_case: Dict) -> str:
        """Generate plan generation prompt"""
        template = self._load_template("plan.txt")
        return self._format_prompt(template, test_case)
    
    def get_code_prompt(self, test_case: Dict, plan: str) -> str:
        """Generate code generation prompt"""
        template = self._load_template("code.txt")
        return self._format_prompt(template, test_case, plan=plan)
    
    def get_direct_prompt(self, test_case: Dict) -> str:
        """Generate direct code generation prompt"""
        template = self._load_template("prompt.txt")
        return self._format_prompt(template, test_case)
    
    def get_cot_prompt(self, test_case: Dict) -> str:
        """Generate Chain-of-Thought prompt"""
        template = self._load_template("prompt.txt")
        return self._format_prompt(template, test_case)

    def get_agent_prompt(self, test_case: Dict) -> str:
        """Generate adaptive-agent prompt."""
        template = self._load_template("prompt.txt")
        return self._format_prompt(template, test_case)
    
    def get_fix_syntax_error_prompt(self, test_case: Dict, code: str, syntax_error: str) -> str:
        """Generate syntax error fix prompt"""
        template = self._load_template("fix_syntax_errors.txt")
        kwargs = {"code": code, "syntax_error": syntax_error}
        return self._format_prompt(template, test_case, **kwargs)


class CodeExecutor:
    """Handles code execution for different solvers"""
    
    def __init__(self, temp_cache_dir: str):
        self.temp_cache_dir = temp_cache_dir
    
    def execute_python_code(self, code: str, code_type: str = "Python", timeout: int = 30) -> Tuple[bool, str]:
        """Execute Python code and return the results.
        
        Args:
            code: The Python code to execute
            code_type: Type of code for error messages (e.g., "Z3", "CSP")
            timeout: Timeout in seconds
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp_filename = tmp.name
            tmp.write(code)

        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run([sys.executable, tmp_filename],
                                   capture_output=True, text=True, timeout=timeout,
                                   env=env)
            os.unlink(tmp_filename)
            
            if result.returncode != 0:
                error_details = f"Stderr: {result.stderr}\nStdout: {result.stdout}"
                return False, f"{code_type} execution error (return code {result.returncode}).\n{error_details}"

            output = result.stdout.strip()
            
            # Optional error checking in output
            # if check_output_errors and ("error" in output.lower() or "exception" in output.lower() or "traceback" in output.lower()):
                # return False, f"{code_type} execution potentially failed:\nOutput:\n```\n{output}\n```\nStderr:\n```\n{result.stderr}\n```"

            return True, output
        except subprocess.TimeoutExpired:
            os.unlink(tmp_filename)
            return False, f"Timeout ({timeout}s) while running {code_type} code. The problem or generated code may be too complex or incorrect."
        except Exception as e:
            if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
                 os.unlink(tmp_filename)
            return False, f"Error executing {code_type} code: {str(e)}"

    def execute_z3_code(self, z3_code: str) -> Tuple[bool, str]:
        """Execute the Python Z3 code and return the results."""
        return self.execute_python_code(z3_code, "Z3", timeout=30)
        
    def execute_pyke_code(self, pyke_code: str) -> Tuple[bool, str]:
        """Execute the PyKe code and return the results."""
        try:
            facts = re.search(r"```facts\n(.*?)```", pyke_code, re.DOTALL).group(1)
            rules = re.search(r"```rules\n(.*?)```", pyke_code, re.DOTALL).group(1)
            query = re.search(r"```query\n(.*?)```", pyke_code, re.DOTALL).group(1)

            if "True" in query:
                final_answer = True
                query = query.replace("True", "$target")
            elif "False" in query:
                final_answer = False
                query = query.replace("False", "$target")
            else:
                raise ValueError("Boolean value for the query target not found")
            
            os.makedirs(self.temp_cache_dir, exist_ok=True)
            with open(os.path.join(self.temp_cache_dir, "facts.kfb"), 'w') as fp:
                fp.write(facts)
            with open(os.path.join(self.temp_cache_dir, "rules.krb"), 'w') as fp:
                fp.write(rules)

            engine = knowledge_engine.engine(self.temp_cache_dir)
            engine.reset()
            engine.activate('rules')
            engine.get_kb('facts')

            answer_list = []
            with engine.prove_goal(query.strip()) as gen:
                found = False
                for vars, plan in gen:
                    found = True
                    answer_list.append(vars['target'])
                if not found:
                    return True, "Unknown"
                else:
                    if True in answer_list and False in answer_list:
                        return True, "multiple answers"
                    else:
                        return True, str(answer_list[0] == final_answer)
                    
        except Exception as e:
            return False, f"Error executing PyKe Program: {str(e)}"
        
        finally:
            if os.path.exists("./compiled_krb"):
                print('removing compiled_krb')
                os.system(f'rm -rf compiled_krb/*')
    
    def execute_csp_code(self, csp_code: str) -> Tuple[bool, str]:
        """Execute the Python CSP code and return the results."""
        return self.execute_python_code(csp_code, "CSP", timeout=20)
    
    def get_execute_function(self, solver_name: str) -> Callable[[str], Tuple[bool, str]]:
        """Get the execution function for a solver"""
        solver_map = {
            'z3': self.execute_z3_code,
            'pyke': self.execute_pyke_code,
            'pythonconstraint': self.execute_csp_code
        }
        return solver_map.get(solver_name)


class CodeCleaner:
    """Handles code cleaning for different datasets
    Note: This is not counted as syntax errors.
    The code cleaning is used to remove the markdown fences and the code blocks from the model output.
    """
    
    @staticmethod
    def clean_code(code_text: str, dataset: str) -> str:
        """Clean the code from the model output which have '```python' or '```' fences"""
        dataset = dataset.lower()
        
        if dataset in ['ar-lsat', 'logicaldeduction', 'proverqa']:
            cleaned_code = code_text.strip()

            # Prefer extracting the first fenced code block when both fences exist.
            fenced_match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n?```", cleaned_code, re.DOTALL)
            if fenced_match:
                return fenced_match.group(1).strip()

            # If the model emitted only a leading fence, strip it anyway.
            cleaned_code = re.sub(r"^\s*```(?:[a-zA-Z0-9_+-]+)?\s*\n?", "", cleaned_code, count=1)

            # If the model emitted only a trailing fence, strip it as well.
            cleaned_code = re.sub(r"\n?\s*```\s*$", "", cleaned_code, count=1)

            return cleaned_code.strip()
        
        elif dataset == 'proofwriter':
            return code_text
        
        elif dataset == 'folio':
            matches = re.search(r"```prover9\n(.*?)```", code_text, re.DOTALL)
            if matches:
                return matches.group(1)
            else:
                print("Warning: No ```prover9 code block found in the output text.")
                return code_text
        
        elif dataset == 'prontoqa':
            return code_text
        
        else:
            raise ValueError(f"Dataset {dataset} not configured for code cleaning.")


class Reasoner(ABC):
    """Base class for different reasoning approaches"""
    
    def __init__(self, 
                 config: ReasonerConfig,
                 data_loader: DataLoader,
                 answer_extractor: AnswerExtractor):
        """Initialize the reasoner with the given components"""
        
        self.config = config
        self.data_loader = data_loader
        self.results_folder = self.create_results_folder()
        self.answer_extractor = answer_extractor
        
        self.summary_folder = os.path.join(self.results_folder, "summary")
        self.log_folder = os.path.join(self.results_folder, "log")
        os.makedirs(self.summary_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)
        
        # Initialize temp cache directory for PyKe
        self.temp_cache_dir = os.path.join(self.results_folder, "temp_cache_dir")
        if os.path.exists("./compiled_krb"):
            print('removing compiled_krb')
            os.system(f'rm -rf ./compiled_krb')

        # Initialize helper classes
        self.prompt_handler = PromptHandler(config.prompt_path, config.dataset)
        self.code_executor = CodeExecutor(self.temp_cache_dir)
        self._initialize_api_clients()

    def __getstate__(self) -> Dict[str, Any]:
        """Drop non-picklable API clients when multiprocessing uses spawn."""
        state = self.__dict__.copy()
        state.pop("api_client", None)
        state.pop("code_api_client", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and recreate API clients inside worker processes."""
        self.__dict__.update(state)
        self._initialize_api_clients()

    def _initialize_api_clients(self) -> None:
        """Initialize API clients based on reasoning method."""
        # Reset to avoid stale references when rehydrating from pickle.
        self.api_client = None
        self.code_api_client = None

        if self.config.reasoning_method in ("cot", "one-step", "adaptive-agent"):
            model = getattr(self.config, "model", None)
            api_config = APIConfig(
                model_name=model,
                temperature=self.config.code_temp,
                gemini_thinking_budget=getattr(self.config, "gemini_thinking_budget", 0),
                gemini_thinking_level=getattr(self.config, "gemini_thinking_level", None),
                openai_compatible_extra_body=self._get_openai_compatible_extra_body(model),
                openai_compatible_timeout_sec=getattr(self.config, "openai_compatible_timeout_sec", 60),
                openai_compatible_max_tokens=getattr(self.config, "openai_compatible_max_tokens", None),
                max_retries=self.config.max_retries,
                inter_test_case_delay=self.config.test_delay
            )
            self.api_client = self.initialize_api_client(model, api_config)
            return

        if self.config.reasoning_method in ("two-step", "three-step"):
            plan_model = getattr(self.config, "plan_model", None)
            print(f"Plan model: {plan_model}")
            if plan_model is None:
                raise ValueError("plan_model is not set")
            plan_api_config = APIConfig(
                model_name=plan_model,
                temperature=self.config.plan_temp,
                gemini_thinking_budget=getattr(self.config, "gemini_thinking_budget", 0),
                gemini_thinking_level=getattr(self.config, "gemini_thinking_level", None),
                openai_compatible_extra_body=self._get_openai_compatible_extra_body(plan_model),
                openai_compatible_timeout_sec=getattr(self.config, "openai_compatible_timeout_sec", 60),
                openai_compatible_max_tokens=getattr(self.config, "openai_compatible_max_tokens", None),
                max_retries=self.config.max_retries,
                inter_test_case_delay=self.config.test_delay
            )
            self.api_client = self.initialize_api_client(plan_model, plan_api_config)

            code_model = getattr(self.config, "code_model", None)
            print(f"Code model: {code_model}")
            if code_model is None:
                raise ValueError("code_model is not set")
            code_api_config = APIConfig(
                model_name=code_model,
                temperature=self.config.code_temp,
                gemini_thinking_budget=getattr(self.config, "gemini_thinking_budget", 0),
                gemini_thinking_level=getattr(self.config, "gemini_thinking_level", None),
                openai_compatible_extra_body=self._get_openai_compatible_extra_body(code_model),
                openai_compatible_timeout_sec=getattr(self.config, "openai_compatible_timeout_sec", 60),
                openai_compatible_max_tokens=getattr(self.config, "openai_compatible_max_tokens", None),
                max_retries=self.config.max_retries,
                inter_test_case_delay=self.config.test_delay
            )
            self.code_api_client = self.initialize_api_client(code_model, code_api_config)

    @staticmethod
    def _normalize_provider_name(provider_name: Optional[str]) -> Optional[str]:
        """Normalize provider aliases to canonical names."""
        if provider_name is None:
            return None
        normalized = provider_name.strip().lower().replace("_", "-")
        if normalized in ("azure", "azure-openai"):
            return "azure-openai"
        if normalized == "gemini":
            return "gemini"
        if normalized == "openai-compatible":
            return "openai-compatible"
        if normalized in ("anthropic", "claude"):
            return "anthropic"
        return normalized

    def _resolve_provider_for_model(self, model_name: str) -> str:
        """Resolve provider by model override, then global provider, then legacy auto-detection."""
        model_providers = getattr(self.config, "model_providers", None)
        if isinstance(model_providers, dict) and model_name in model_providers:
            provider = self._normalize_provider_name(model_providers[model_name])
            if provider:
                return provider

        global_provider = self._normalize_provider_name(getattr(self.config, "provider", None))
        if global_provider:
            return global_provider

        model_name_lower = model_name.lower()
        if 'gpt' in model_name_lower:
            return "azure-openai"
        if 'gemini' in model_name_lower:
            return "gemini"
        if 'claude' in model_name_lower:
            return "anthropic"
        raise ValueError(
            f"Unable to auto-detect provider for model '{model_name}'. "
            "Set config.provider or config.model_providers in YAML."
        )

    def _get_openai_compatible_extra_body(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Fetch per-model OpenAI-compatible extra_body from config."""
        per_model = getattr(self.config, "openai_compatible_extra_body", None)
        if isinstance(per_model, dict):
            model_extra_body = per_model.get(model_name)
            if isinstance(model_extra_body, dict):
                return model_extra_body

        default_extra_body = getattr(self.config, "openai_compatible_default_extra_body", None)
        if isinstance(default_extra_body, dict):
            return default_extra_body
        return None

    def initialize_api_client(self, model_name: str, api_config: APIConfig) -> APIClient:
        """Initialize the API client for the given model name and API configuration"""
        client_params = {}
        provider = self._resolve_provider_for_model(model_name)
        api_config.provider = provider

        if provider == "azure-openai":
            client_params = {
                'endpoint': self.config.azure_endpoint,
                'deployment': self.config.azure_deployment,
                'managed_identity_client_id': self.config.azure_managed_identity_client_id
            }
        elif provider == "gemini":
            if self.config.gemini_api_keys:
                client_params = {'api_keys': self.config.gemini_api_keys}
                print(f"Initialized Gemini client with {len(self.config.gemini_api_keys)} API keys for rotation")
            else:
                single_key = getattr(self.config, "gemini_api_key", None) or getattr(self.config, "api_key", None)
                client_params = {'api_key': single_key}
        elif provider == "openai-compatible":
            openai_compatible_api_key = getattr(self.config, "openai_compatible_api_key", None) or getattr(self.config, "api_key", None)
            client_params = {
                'api_key': openai_compatible_api_key,
                'base_url': getattr(self.config, "openai_compatible_base_url", None),
            }
        elif provider == "anthropic":
            anthropic_api_key = getattr(self.config, "anthropic_api_key", None) or getattr(self.config, "api_key", None)
            client_params = {
                'api_key': anthropic_api_key,
                'base_url': getattr(self.config, "anthropic_base_url", None),
            }
        else:
            raise ValueError(f"Unsupported provider '{provider}' for model '{model_name}'")
        
        api_client = get_api_client(api_config.provider, api_config, **client_params)
        return api_client
    
    def save_prompts_folder(self) -> None:
        """Copy the prompts folder to the results directory to preserve exact prompts used"""
        if not os.path.exists(self.config.prompt_path):
            print(f"Warning: Prompt path {self.config.prompt_path} does not exist, skipping prompt folder copy")
            return
            
        prompts_dest = os.path.join(self.results_folder, "prompts")
        
        try:
            if os.path.exists(prompts_dest):
                print(f"Prompts folder already exists at {prompts_dest}, removing old copy...")
                shutil.rmtree(prompts_dest)
            
            shutil.copytree(self.config.prompt_path, prompts_dest)
            print(f"Prompts folder copied to: {prompts_dest}")
            
            prompt_metadata = {
                "original_prompt_path": self.config.prompt_path,
                "copied_at": datetime.now().isoformat(),
                "reasoning_method": self.config.reasoning_method,
                "dataset": self.config.dataset,
                "shots": self.config.shots
            }
            
            metadata_path = os.path.join(prompts_dest, "prompt_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(prompt_metadata, f, indent=2)
            
            print(f"Prompt metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"Error copying prompts folder: {e}")
            print(f"   Source: {self.config.prompt_path}")
            print(f"   Destination: {prompts_dest}")

    @abstractmethod
    def reason(self, test_case: Dict) -> Dict:
        """Implement the reasoning strategy"""
        pass

    def create_results_folder(self) -> None:
        """Create results folder based on model name"""
        results_root = getattr(self.config, "results_root", "./results")
        if self.config.reasoning_method in ("cot", "one-step", "adaptive-agent"):
            model_name = getattr(self.config, "model", None)
            if model_name is None:
                raise ValueError("model is not set")
            safe_model_name = self._sanitize_name_for_path(model_name)
            folder_name = (
                f"{self.config.reasoning_method}-{self.config.dataset}-model-{safe_model_name}-"
                f"{self.config.shots}_shot-{str(uuid.uuid4())}"
            )
        else:
            plan_model_name = getattr(self.config, 'plan_model', None)
            if plan_model_name is None:
                raise ValueError("plan_model is not set")
            code_model_name = getattr(self.config, 'code_model', None)
            if code_model_name is None:
                raise ValueError("code_model is not set")
            safe_plan_model_name = self._sanitize_name_for_path(plan_model_name)
            safe_code_model_name = self._sanitize_name_for_path(code_model_name)
            folder_name = (
                f"{self.config.reasoning_method}-{self.config.dataset}-plan-with-{safe_plan_model_name}-"
                f"code-with-{safe_code_model_name}-{self.config.shots}_shot_CoT-{str(uuid.uuid4())}"
            )

        self.results_folder = os.path.join(
            results_root,
            f"results_{datetime.now().strftime('%Y-%m-%d')}",
            folder_name
        )
        print(f"Results folder: {self.results_folder}")
        
        if os.path.exists(self.results_folder):
            print(f"The results folder {self.results_folder} already exists, check whether you want to continue")
        else:
            print("No existing results found, starting fresh")
            os.makedirs(self.results_folder, exist_ok=True)
        return self.results_folder

    def _call_api(self, prompt: str) -> str:
        """Common method to call the API using the modular client"""
        return self.api_client.call(prompt)
    
    def _call_fix_api(self, prompt: str) -> str:
        """Common method to call the API using the modular client"""
        return self.code_api_client.call(prompt)

    def _call_api_with_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call a chat endpoint with structured messages and optional tools."""
        if not hasattr(self.api_client, "call_with_messages"):
            raise ValueError("The configured API client does not support structured message/tool calls.")
        return self.api_client.call_with_messages(messages, tools=tools, tool_choice=tool_choice)
    
    def interpret_results(self, response_text: str) -> Tuple[bool, str, Optional[str]]:
        """Interpret the results from the reasoning"""
        return True, "Model passed the test.", response_text

    def _process_results(self, test_case: Dict, reasoning_result: Dict, case_time: float, unique_id: str="", timing_data: Optional[Dict] = None) -> None:
        """Process the results of a single test case"""
        pass

    @staticmethod
    def _sanitize_name_for_path(name: str) -> str:
        """Convert names into filesystem-safe path segments."""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    @staticmethod
    def _zero_token_usage() -> Dict[str, int]:
        """Create a zeroed token usage dict."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @classmethod
    def _normalize_token_usage(cls, usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Normalize possibly-null token usage into integer totals."""
        if not isinstance(usage, dict):
            return cls._zero_token_usage()
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    @classmethod
    def _add_token_usage(cls, accum: Dict[str, int], usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """Add token usage into an accumulator and return the updated dict."""
        normalized = cls._normalize_token_usage(usage)
        return {
            "prompt_tokens": accum["prompt_tokens"] + normalized["prompt_tokens"],
            "completion_tokens": accum["completion_tokens"] + normalized["completion_tokens"],
            "total_tokens": accum["total_tokens"] + normalized["total_tokens"],
        }

    @classmethod
    def _sum_token_usages(cls, usage_items: List[Optional[Dict[str, Any]]]) -> Dict[str, int]:
        """Sum a list of token usage dicts."""
        total = cls._zero_token_usage()
        for usage in usage_items:
            total = cls._add_token_usage(total, usage)
        return total

    @staticmethod
    def _zero_api_time() -> float:
        """Create a zero value for api_time_only (seconds)."""
        return 0.0

    @staticmethod
    def _zero_solver_time() -> float:
        """Create a zero value for solver_time_only (seconds)."""
        return 0.0

    @classmethod
    def _normalize_api_time(cls, api_time_only: Optional[Any]) -> float:
        """Normalize api_time_only to a non-negative float."""
        try:
            value = float(api_time_only)
            return value if value >= 0 else 0.0
        except Exception:
            return 0.0

    @classmethod
    def _sum_api_times(cls, api_times: List[Optional[Any]]) -> float:
        """Sum a list of api_time_only values."""
        total = 0.0
        for api_time in api_times:
            total += cls._normalize_api_time(api_time)
        return total

    @classmethod
    def _normalize_solver_time(cls, solver_time_only: Optional[Any]) -> float:
        """Normalize solver_time_only to a non-negative float."""
        return cls._normalize_api_time(solver_time_only)

    @classmethod
    def _sum_solver_times(cls, solver_times: List[Optional[Any]]) -> float:
        """Sum a list of solver_time_only values."""
        total = 0.0
        for solver_time in solver_times:
            total += cls._normalize_solver_time(solver_time)
        return total

    def _extract_usage(self, client: APIClient) -> Dict[str, Optional[int]]:
        """Get normalized token usage fields from the latest API call metadata."""
        metadata = client.get_last_call_metadata() if hasattr(client, "get_last_call_metadata") else {}
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
        if not isinstance(usage, dict):
            return {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    def _extract_api_time(self, client: APIClient) -> float:
        """Get api_time_only (seconds) from the latest API call metadata."""
        metadata = client.get_last_call_metadata() if hasattr(client, "get_last_call_metadata") else {}
        timing = metadata.get("timing") if isinstance(metadata, dict) else None
        if not isinstance(timing, dict):
            return 0.0
        return self._normalize_api_time(timing.get("api_time_only_sec"))

    def _write_run_token_usage_summary(self) -> None:
        """Aggregate token totals across all per-sample summaries for this run."""
        if not os.path.exists(self.summary_folder):
            return

        total_usage = self._zero_token_usage()
        total_api_time_only = self._zero_api_time()
        total_solver_time_only = self._zero_solver_time()
        samples_with_usage = 0
        samples_with_api_time = 0
        samples_with_solver_time = 0
        total_samples = 0

        for filename in os.listdir(self.summary_folder):
            if not filename.endswith(".json"):
                continue
            total_samples += 1
            summary_path = os.path.join(self.summary_folder, filename)
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception:
                continue

            sample_usage = summary_data.get("token_usage_total")
            if sample_usage is not None:
                samples_with_usage += 1
            total_usage = self._add_token_usage(total_usage, sample_usage)
            sample_api_time = summary_data.get("api_time_only_total")
            if sample_api_time is not None:
                samples_with_api_time += 1
            total_api_time_only += self._normalize_api_time(sample_api_time)
            sample_solver_time = summary_data.get("solver_time_only_total")
            if sample_solver_time is not None:
                samples_with_solver_time += 1
            total_solver_time_only += self._normalize_solver_time(sample_solver_time)

        output_path = os.path.join(self.results_folder, "token_usage_summary.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_samples": total_samples,
                "samples_with_usage": samples_with_usage,
                "token_usage_total": total_usage,
                "samples_with_api_time_only": samples_with_api_time,
                "api_time_only_total": total_api_time_only,
                "samples_with_solver_time_only": samples_with_solver_time,
                "solver_time_only_total": total_solver_time_only,
            }, f, indent=2, ensure_ascii=False)

    def run_all_tests(self) -> None:
        """Run reasoning on all test cases in the file with optional limit"""
        start_time_total = time.time()
        processed_count = 0
        for i, batch in enumerate(self.data_loader):
            case_start_time = time.time()
            reasoning_result = self.reason(batch[0])
            case_time = time.time() - case_start_time
            self._process_results(batch[0], reasoning_result, case_time, str(uuid.uuid4()), None)

            processed_count += 1
            if processed_count < len(self.data_loader):
                print(f"Waiting {self.config.test_delay}s before next test case...")
                time.sleep(self.config.test_delay)

        total_execution_time = time.time() - start_time_total
        print(f"\nTotal execution time: {total_execution_time:.2f}s")
        with open(os.path.join(self.results_folder, "total_execution_time.txt"), "w", encoding="utf-8") as f:
            f.write(f"{total_execution_time:.2f}s")
        self._write_run_token_usage_summary()
        
    def run_all_tests_parallel(self, num_processes: int=10) -> None:
        """Use multiple processes to run all test cases in parallel"""
        print(f"Start parallel testing, using {num_processes} processes...")
        print(f"Please visit the {self.log_folder} to view the real-time output log")

        start_time_total = time.time()
        
        mp_lock = mp.Lock()
        all_tasks = [(self, batch[0], mp_lock, idx) for idx, batch in enumerate(self.data_loader)]
        processes = []
        try:
            for task in all_tasks:
                while len(processes) >= num_processes:
                    processes = [p for p in processes if p.is_alive()]
                    time.sleep(0.1)
                p = mp.Process(target=_parallel_worker, args=(task,))
                p.start()
                processes.append(p)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received! Terminating all processes...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
        finally:
            for p in processes:
                p.join()
            print("All processes joined.")

        total_execution_time = time.time() - start_time_total
        print(f"\nTotal execution time: {total_execution_time:.2f}s")
        with open(os.path.join(self.results_folder, "total_execution_time.txt"), "w", encoding="utf-8") as f:
            f.write(f"{total_execution_time:.2f}s")
        self._write_run_token_usage_summary()


class CodeBasedReasoner(Reasoner):
    """Base class for reasoners that generate and execute code"""
    
    def reason_code_diversity(self, 
                              test_case: dict,
                              solver_name: str,
                              mp_lock: Optional[Any] = None) -> dict:
        """Generate and execute code with diversity parameters"""
        execute_func = self.code_executor.get_execute_function(solver_name)
        if execute_func is None:
            raise ValueError(f"Unsupported solver: {solver_name}")
        
        return self._generate_and_execute_code(test_case, solver_name, execute_func, mp_lock)
    
    @abstractmethod
    def _generate_and_execute_code(self, test_case: dict, solver_name: str, execute_func: Callable, mp_lock: Optional[Any] = None) -> dict:
        """Generate and execute code - to be implemented by subclasses"""
        pass

    def _execute_code(self, execute_func: Callable, code: str, mp_lock: Optional[Any]) -> Tuple[bool, str, float]:
        """Execute solver code with optional multiprocessing lock."""
        if mp_lock is not None:
            with mp_lock:
                solver_start_time = time.time()
                is_valid, solver_output = execute_func(code)
                solver_time_only = time.time() - solver_start_time
                return is_valid, solver_output, solver_time_only
        solver_start_time = time.time()
        is_valid, solver_output = execute_func(code)
        solver_time_only = time.time() - solver_start_time
        return is_valid, solver_output, solver_time_only

    @staticmethod
    def _classify_solver_error(output: Any) -> Optional[str]:
        """Classify solver failures into syntax_error, solver_timeout, or runtime_error."""
        if output is None:
            return None
        if not isinstance(output, str):
            return "runtime_error"

        lower = output.lower()
        if "timeout" in lower:
            return "solver_timeout"

        syntax_markers = [
            "syntaxerror",
            "indentationerror",
            "taberror",
            "invalid syntax",
            "eol while scanning string literal",
            "unexpected eof while parsing",
        ]
        if any(marker in lower for marker in syntax_markers):
            return "syntax_error"

        if "error" in lower or "exception" in lower or "traceback" in lower:
            return "runtime_error"

        return None
    
    def _execute_with_repair(self, test_case: dict, code: str, execute_func: Callable,
                           mp_lock: Optional[Any], identifier: str,
                           prompt_handler: Optional[PromptHandler] = None,
                           repair_client: Optional[APIClient] = None) -> Tuple[str, str, bool, List[Dict[str, Any]], Optional[str], float]:
        """Execute code with repair attempts"""
        temp_code = code
        repair_round_logs: List[Dict[str, Any]] = []
        solver_time_only_total = self._zero_solver_time()
        active_prompt_handler = prompt_handler or self.prompt_handler

        print(f"Starting syntax error iteration 1/{self.config.max_repairs} for {identifier}")
        is_valid, temp_solver_output, initial_solver_time_only = self._execute_code(execute_func, temp_code, mp_lock)
        solver_time_only_total += initial_solver_time_only

        # Keep existing behavior: max_repair loop allows up to (max_repairs - 1) repair attempts.
        for repair_round in range(1, self.config.max_repairs):
            if is_valid:
                print(f"Code execution succeeded for {identifier}.")
                break

            print(f"Code execution failed for {identifier}. Error: {temp_solver_output}")
            fix_prompt = active_prompt_handler.get_fix_syntax_error_prompt(test_case, temp_code, temp_solver_output)

            if repair_client is not None:
                fix_response = repair_client.call(fix_prompt)
                usage = self._extract_usage(repair_client)
                api_time_only = self._extract_api_time(repair_client)
            elif hasattr(self, 'code_api_client'):
                fix_response = self._call_fix_api(fix_prompt)
                usage = self._extract_usage(self.code_api_client)
                api_time_only = self._extract_api_time(self.code_api_client)
            else:
                fix_response = self._call_api(fix_prompt)
                usage = self._extract_usage(self.api_client)
                api_time_only = self._extract_api_time(self.api_client)

            repaired_code = CodeCleaner.clean_code(fix_response, self.config.dataset)

            print(f"Starting syntax error iteration {repair_round + 1}/{self.config.max_repairs} for {identifier}")
            is_valid, temp_solver_output, solver_time_only = self._execute_code(execute_func, repaired_code, mp_lock)
            solver_time_only_total += solver_time_only

            error_type = None if is_valid else self._classify_solver_error(temp_solver_output)
            repair_round_logs.append({
                "round_index": repair_round,
                "input_code": temp_code,
                "repair_prompt": fix_prompt,
                "repaired_code": repaired_code,
                "syntax_check_result": {
                    "is_valid": is_valid,
                    "solver_output_or_error": temp_solver_output,
                    "error_type": error_type,
                },
                "usage": usage,
                "api_time_only": api_time_only,
                "solver_time_only": solver_time_only,
            })

            temp_code = repaired_code

        final_error_type = None if is_valid else self._classify_solver_error(temp_solver_output)
        return temp_code, temp_solver_output, is_valid, repair_round_logs, final_error_type, solver_time_only_total
    
    def reason(self, test_case: Dict, mp_lock: Optional[Any]=None) -> Dict:
        """Main reasoning entry point"""
        solver_name = DatasetConfig.get_solver(self.config.dataset)
        if solver_name is None:
            raise ValueError(f"Dataset {self.config.dataset} not configured for reasoning.")
        return self.reason_code_diversity(test_case, solver_name, mp_lock)


class TwoStepReasoner(CodeBasedReasoner):
    """Two-step reasoning approach"""

    def _generate_and_execute_code(self, test_case: dict, solver_name: str, execute_func: Callable, mp_lock: Optional[Any] = None) -> dict:
        """Generate plans and codes with diversity parameters"""
        plan_configs = [{"temperature": self.config.plan_temp} for _ in range(self.config.num_paths)]
        all_plan_results = []
        
        for plan_config_idx, plan_config in enumerate(plan_configs):
            print(f"Generating plan {plan_config_idx + 1}/{len(plan_configs)}...")
            self.api_client.temperature = plan_config["temperature"]
            
            # Generate plan
            plan_prompt = self.prompt_handler.get_plan_prompt(test_case)
            current_plan_response = self._call_api(plan_prompt)
            current_plan = current_plan_response if isinstance(current_plan_response, str) else current_plan_response[0]
            plan_usage = self._extract_usage(self.api_client)
            plan_api_time_only = self._extract_api_time(self.api_client)
            
            # Generate code
            code_configs = [{"temperature": self.config.code_temp}]
            plan_code_results = []
            
            for code_gen_idx, code_config in enumerate(code_configs, 1):
                self.code_api_client.temperature = code_config["temperature"]
                
                code_prompt = self.prompt_handler.get_code_prompt(test_case, current_plan)
                temp_code_response = self._call_fix_api(code_prompt)
                code_generation_usage = self._extract_usage(self.code_api_client)
                code_generation_api_time_only = self._extract_api_time(self.code_api_client)
                temp_code = CodeCleaner.clean_code(temp_code_response, self.config.dataset)
                
                # Execute with repair loop
                identifier = f"plan={plan_config_idx + 1}, code={code_gen_idx}"
                temp_code, temp_solver_output, is_valid, repair_round_logs, final_error_type, solver_time_only = self._execute_with_repair(
                    test_case, temp_code, execute_func, mp_lock, identifier
                )
                repair_usage_total = self._sum_token_usages([log.get("usage") for log in repair_round_logs])
                repair_api_time_only_total = self._sum_api_times([log.get("api_time_only") for log in repair_round_logs])
                path_usage_total = self._sum_token_usages([plan_usage, code_generation_usage, repair_usage_total])
                path_api_time_only = self._sum_api_times([plan_api_time_only, code_generation_api_time_only, repair_api_time_only_total])
                
                code_result = {
                    "plan_idx": plan_config_idx + 1,
                    "code_idx": code_gen_idx,
                    "code": temp_code,
                    "solver_output": temp_solver_output,
                    "is_valid": is_valid,
                    "code_generation_config": code_config.copy(),
                    "plan_generation_usage": plan_usage,
                    "plan_generation_api_time_only": plan_api_time_only,
                    "code_generation_usage": code_generation_usage,
                    "code_generation_api_time_only": code_generation_api_time_only,
                    "repair_usage_total": repair_usage_total,
                    "repair_api_time_only_total": repair_api_time_only_total,
                    "path_usage_total": path_usage_total,
                    "path_api_time_only": path_api_time_only,
                    "solver_time_only": solver_time_only,
                    "repair_round_logs": repair_round_logs,
                    "solver_error_type": final_error_type,
                }
                plan_code_results.append(code_result)
            
            plan_result = {
                "plan_idx": plan_config_idx + 1,
                "plan_config": plan_config.copy(),
                "plan": current_plan,
                "plan_generation_usage": plan_usage,
                "plan_generation_api_time_only": plan_api_time_only,
                "code_results": plan_code_results
            }
            all_plan_results.append(plan_result)
        
        return {"all_plan_results": all_plan_results}

    def _process_results(self, test_case: Dict, reasoning_result: Dict, case_time: float, unique_id: str="", timing_data: Optional[Dict] = None) -> None:
        """Save the plans, codes and results to the results_folder"""
        plan_folder = os.path.join(self.results_folder, "plan")
        code_folder = os.path.join(self.results_folder, "code")
        os.makedirs(plan_folder, exist_ok=True)
        os.makedirs(code_folder, exist_ok=True)

        problem_name = test_case['id_string'] if 'id_string' in test_case else test_case['id']
        all_plan_results = reasoning_result.get("all_plan_results", [])
        all_solver_outputs = []
        all_solver_error_types = []
        path_token_usage = []
        token_usage_total = self._zero_token_usage()
        api_time_only_total = self._zero_api_time()
        solver_time_only_total = self._zero_solver_time()
        repair_logs = []
        
        for plan_result in all_plan_results:
            plan_idx = plan_result["plan_idx"]
            plan_config = plan_result["plan_config"]
            plan_temp = plan_config["temperature"]
            
            # Save plan
            plan_filepath = os.path.join(plan_folder, f"{problem_name}-{unique_id}-plan{plan_idx}-temp{plan_temp}.txt")
            with open(plan_filepath, "w") as f:
                # f.write(f"Plan Config: {plan_config}\n\n")
                f.write(plan_result["plan"])
            
            # Save codes
            for code_result in plan_result.get("code_results", []):
                code_idx = code_result["code_idx"]
                code_generation_config = code_result.get("code_generation_config", {})
                
                code_filepath = os.path.join(code_folder, f"{problem_name}-{unique_id}-plan{plan_idx}-code{code_idx}.py")
                with open(code_filepath, "w") as f:
                    # f.write(f"# Plan Config: {plan_config}\n")
                    # f.write(f"# Generation Config: {code_generation_config}\n\n")
                    f.write(code_result["code"])
                
                solver_output = code_result["solver_output"]
                if solver_output is not None:
                    all_solver_outputs.append(solver_output)
                    all_solver_error_types.append(code_result.get("solver_error_type"))
                path_usage_total = self._normalize_token_usage(code_result.get("path_usage_total"))
                token_usage_total = self._add_token_usage(token_usage_total, path_usage_total)
                path_api_time_only = self._normalize_api_time(code_result.get("path_api_time_only"))
                api_time_only_total += path_api_time_only
                path_solver_time_only = self._normalize_solver_time(code_result.get("solver_time_only"))
                solver_time_only_total += path_solver_time_only
                path_token_usage.append({
                    "plan_idx": plan_idx,
                    "code_idx": code_idx,
                    "plan_generation_usage": self._normalize_token_usage(code_result.get("plan_generation_usage")),
                    "plan_generation_api_time_only": self._normalize_api_time(code_result.get("plan_generation_api_time_only")),
                    "code_generation_usage": self._normalize_token_usage(code_result.get("code_generation_usage")),
                    "code_generation_api_time_only": self._normalize_api_time(code_result.get("code_generation_api_time_only")),
                    "repair_usage_total": self._normalize_token_usage(code_result.get("repair_usage_total")),
                    "repair_api_time_only_total": self._normalize_api_time(code_result.get("repair_api_time_only_total")),
                    "path_usage_total": path_usage_total,
                    "path_api_time_only": path_api_time_only,
                    "solver_time_only": path_solver_time_only,
                })
                repair_logs.append({
                    "plan_idx": plan_idx,
                    "code_idx": code_idx,
                    "repair_round_logs": code_result.get("repair_round_logs", [])
                })
            
        results = {
            "problem": test_case,
            "timing": case_time,
            "all_solver_outputs": all_solver_outputs,
            "all_solver_error_types": all_solver_error_types,
            "path_token_usage": path_token_usage,
            "token_usage_total": token_usage_total,
            "api_time_only_total": api_time_only_total,
            "solver_time_only_total": solver_time_only_total,
            "repair_logs": repair_logs,
        }

        summary_filepath = os.path.join(self.summary_folder, f"{problem_name}-{unique_id}.json")
        with open(summary_filepath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


class DirectReasoner(CodeBasedReasoner):
    """Direct reasoning approach - generates formal code in one step"""

    def _generate_and_execute_code(self, test_case: dict, solver_name: str, execute_func: Callable, mp_lock: Optional[Any] = None) -> dict:
        """Generate multiple code variants with diverse parameters"""
        code_configs = [{"temperature": self.config.code_temp} for _ in range(self.config.num_paths)]
        all_code_results = []
        original_temp = self.api_client.temperature
        
        for code_config_idx, code_config in enumerate(code_configs):
            print(f"Generating code {code_config_idx + 1}/{len(code_configs)}...")
            self.api_client.temperature = code_config["temperature"]
            
            # Generate code directly
            direct_prompt = self.prompt_handler.get_direct_prompt(test_case)
            current_code_response = self._call_api(direct_prompt)
            code_generation_usage = self._extract_usage(self.api_client)
            code_generation_api_time_only = self._extract_api_time(self.api_client)
            current_code = current_code_response if isinstance(current_code_response, str) else current_code_response[0]
            current_code = CodeCleaner.clean_code(current_code, self.config.dataset)
            
            # Execute with repair loop
            identifier = f"code {code_config_idx + 1}"
            current_code, temp_solver_output, is_valid, repair_round_logs, final_error_type, solver_time_only = self._execute_with_repair(
                test_case, current_code, execute_func, mp_lock, identifier
            )
            repair_usage_total = self._sum_token_usages([log.get("usage") for log in repair_round_logs])
            repair_api_time_only_total = self._sum_api_times([log.get("api_time_only") for log in repair_round_logs])
            path_usage_total = self._sum_token_usages([code_generation_usage, repair_usage_total])
            path_api_time_only = self._sum_api_times([code_generation_api_time_only, repair_api_time_only_total])
            
            code_result = {
                "code_idx": code_config_idx + 1,
                "code": current_code,
                "solver_output": temp_solver_output,
                "is_valid": is_valid,
                "generation_config": code_config.copy(),
                "code_generation_usage": code_generation_usage,
                "code_generation_api_time_only": code_generation_api_time_only,
                "repair_usage_total": repair_usage_total,
                "repair_api_time_only_total": repair_api_time_only_total,
                "path_usage_total": path_usage_total,
                "path_api_time_only": path_api_time_only,
                "solver_time_only": solver_time_only,
                "repair_round_logs": repair_round_logs,
                "solver_error_type": final_error_type,
            }
            all_code_results.append(code_result)
        
        self.api_client.temperature = original_temp
        return {"all_code_results": all_code_results}

    def _process_results(self, test_case: Dict, reasoning_result: Dict, case_time: float, unique_id: str="", timing_data: Optional[Dict] = None) -> None:
        """Save the codes and results to the results_folder"""
        code_folder = os.path.join(self.results_folder, "code")
        os.makedirs(code_folder, exist_ok=True)

        problem_name = test_case['id_string'] if 'id_string' in test_case else test_case['id']
        all_code_results = reasoning_result.get("all_code_results", [])
        all_solver_outputs = []
        all_solver_error_types = []
        path_token_usage = []
        token_usage_total = self._zero_token_usage()
        api_time_only_total = self._zero_api_time()
        solver_time_only_total = self._zero_solver_time()
        repair_logs = []
        
        for code_result in all_code_results:
            code_idx = code_result["code_idx"]
            code_generation_config = code_result.get("generation_config", {})
            
            code_filepath = os.path.join(code_folder, f"{problem_name}-{unique_id}-code{code_idx}.py")
            with open(code_filepath, "w") as f:
                # f.write(f"# Generation Config: {code_generation_config}\n\n")
                f.write(code_result["code"])
            
            solver_output = code_result["solver_output"]
            if solver_output is not None:
                all_solver_outputs.append(solver_output)
                all_solver_error_types.append(code_result.get("solver_error_type"))
            path_usage_total = self._normalize_token_usage(code_result.get("path_usage_total"))
            token_usage_total = self._add_token_usage(token_usage_total, path_usage_total)
            path_api_time_only = self._normalize_api_time(code_result.get("path_api_time_only"))
            api_time_only_total += path_api_time_only
            path_solver_time_only = self._normalize_solver_time(code_result.get("solver_time_only"))
            solver_time_only_total += path_solver_time_only
            path_token_usage.append({
                "code_idx": code_idx,
                "code_generation_usage": self._normalize_token_usage(code_result.get("code_generation_usage")),
                "code_generation_api_time_only": self._normalize_api_time(code_result.get("code_generation_api_time_only")),
                "repair_usage_total": self._normalize_token_usage(code_result.get("repair_usage_total")),
                "repair_api_time_only_total": self._normalize_api_time(code_result.get("repair_api_time_only_total")),
                "path_usage_total": path_usage_total,
                "path_api_time_only": path_api_time_only,
                "solver_time_only": path_solver_time_only,
            })
            repair_logs.append({
                "code_idx": code_idx,
                "repair_round_logs": code_result.get("repair_round_logs", [])
            })
            
        results = {
            "problem": test_case,
            "timing": case_time,
            "all_solver_outputs": all_solver_outputs,
            "all_solver_error_types": all_solver_error_types,
            "path_token_usage": path_token_usage,
            "token_usage_total": token_usage_total,
            "api_time_only_total": api_time_only_total,
            "solver_time_only_total": solver_time_only_total,
            "repair_logs": repair_logs,
        }
        
        summary_filepath = os.path.join(self.summary_folder, f"{problem_name}-{unique_id}.json")
        with open(summary_filepath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    

class CoTReasoner(Reasoner):
    """Chain-of-Thought reasoning approach"""

    def reason(self, test_case: Dict, mp_lock: Optional[Any]=None) -> Dict:
        """Generate multiple reasoning outputs using the CoT prompt with diverse parameters"""
        cot_configs = [{"temperature": self.config.code_temp} for _ in range(self.config.num_paths)]
        all_reasoning_results = []
        
        for cot_config_idx, cot_config in enumerate(cot_configs):
            print(f"\nGenerating CoT reasoning {cot_config_idx + 1}/{len(cot_configs)}...")
            self.api_client.temperature = cot_config["temperature"]
            
            # Generate CoT reasoning
            cot_prompt = self.prompt_handler.get_cot_prompt(test_case)
            current_reasoning_response = self._call_api(cot_prompt)
            current_reasoning = current_reasoning_response if isinstance(current_reasoning_response, str) else current_reasoning_response[0]
            cot_usage = self._extract_usage(self.api_client)
            cot_api_time_only = self._extract_api_time(self.api_client)
            
            reasoning_result = {
                "reasoning_idx": cot_config_idx + 1,
                "reasoning_output": current_reasoning,
                "usage": cot_usage,
                "api_time_only": cot_api_time_only,
                "generation_config": cot_config.copy()
            }
            all_reasoning_results.append(reasoning_result)
        
        return {"all_reasoning_results": all_reasoning_results}

    def _process_results(self, test_case: Dict, reasoning_result: Dict, case_time: float, unique_id: str="", timing_data: Optional[Dict] = None) -> None:
        """Save the reasoning outputs and extract answers from all paths"""
        reasoning_folder = os.path.join(self.results_folder, "reasoning")
        os.makedirs(reasoning_folder, exist_ok=True)

        problem_name = test_case['id_string'] if 'id_string' in test_case else test_case['id']
        all_reasoning_results = reasoning_result.get("all_reasoning_results", [])
        all_reasoning_outputs = []
        all_solver_outputs = []
        path_token_usage = []
        token_usage_total = self._zero_token_usage()
        api_time_only_total = self._zero_api_time()
        solver_time_only_total = self._zero_solver_time()
        
        for reasoning_result_item in all_reasoning_results:
            reasoning_idx = reasoning_result_item["reasoning_idx"]
            reasoning_output = reasoning_result_item.get("reasoning_output", "")
            if reasoning_output is None:
                reasoning_output = "[ERROR: No reasoning output generated]"
            
            # Save each reasoning output to a separate file
            reasoning_filepath = os.path.join(reasoning_folder, f"{problem_name}-{unique_id}-reasoning{reasoning_idx}.txt")
            with open(reasoning_filepath, "w") as f:
                f.write(reasoning_output)
            
            # Extract answer from this reasoning output
            reasoning_output_for_extraction = reasoning_output if reasoning_output != "[ERROR: No reasoning output generated]" else ""
            
            # Dataset-specific answer extraction for correctness check and answer index
            if self.config.dataset.lower() == "ar-lsat":
                is_correct, extracted_answer_index, error_type = self.answer_extractor.extract_answer(
                    reasoning_output_for_extraction, test_case["label"], self.config.reasoning_method)
            elif self.config.dataset.lower() in ["proofwriter", "folio", "proverqa", "prontoqa", "logicaldeduction"]:
                is_correct, extracted_answer_index, error_type = self.answer_extractor.extract_answer(
                    reasoning_output_for_extraction, test_case["answer"], self.config.reasoning_method)
            else:
                raise ValueError(f"Dataset {self.config.dataset} not configured for CoTReasoner AnswerExtractor.")
            
            # Handle API failure case
            if reasoning_output == "[ERROR: No reasoning output generated]":
                is_correct = False
                error_type = "API failure - no output generated"
                extracted_answer_index = "[]"
            
            # Add extracted answer index to all_solver_outputs
            all_solver_outputs.append(extracted_answer_index)
            
            all_reasoning_outputs.append({
                "reasoning_idx": reasoning_idx,
                "reasoning_output": reasoning_output,
                "error_type": error_type,
                "success": is_correct,
                "usage": self._normalize_token_usage(reasoning_result_item.get("usage")),
                "api_time_only": self._normalize_api_time(reasoning_result_item.get("api_time_only")),
                "generation_config": reasoning_result_item.get("generation_config", {})
            })
            path_usage_total = self._normalize_token_usage(reasoning_result_item.get("usage"))
            token_usage_total = self._add_token_usage(token_usage_total, path_usage_total)
            path_api_time_only = self._normalize_api_time(reasoning_result_item.get("api_time_only"))
            api_time_only_total += path_api_time_only
            path_token_usage.append({
                "reasoning_idx": reasoning_idx,
                "path_usage_total": path_usage_total,
                "path_api_time_only": path_api_time_only,
                "solver_time_only": self._zero_solver_time(),
            })
            
            print(f"\nReasoning {reasoning_idx} {'PASSED' if is_correct else 'FAILED'}. Error type: {error_type}")

        results = {
            "problem": test_case,
            "timing": case_time,
            "all_reasoning_outputs": all_reasoning_outputs,
            "all_solver_outputs": all_solver_outputs,
            "path_token_usage": path_token_usage,
            "token_usage_total": token_usage_total,
            "api_time_only_total": api_time_only_total,
            "solver_time_only_total": solver_time_only_total,
        }
        
        summary_filepath = os.path.join(self.summary_folder, f"{problem_name}-{unique_id}.json")
        with open(summary_filepath, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


class AdaptiveAgentReasoner(CodeBasedReasoner):
    """Adaptive reasoning that lets the model decide whether to call Z3."""

    MAX_AGENT_STEPS = 6
    MAX_Z3_TOOL_CALLS = 3

    DATASET_SETTINGS = {
        "ar-lsat": {
            "tool_prompt_dir": "AR-LSAT-prompts-one-step",
            "tool_name": "solve_ar_lsat_with_z3",
            "choices_key": "answers",
            "expected_choice_count": 5,
            "tool_description": (
                "Use symbolic reasoning with Z3 to solve the AR-LSAT problem. "
                "Call this when the constraints are easier to verify formally than by direct reasoning."
            ),
            "context_description": "The full AR-LSAT problem context.",
            "question_description": "The AR-LSAT question to answer.",
        },
        "proverqa": {
            "tool_prompt_dir": "ProverQA-prompts-one-step",
            "tool_name": "solve_proverqa_with_z3",
            "choices_key": "options",
            "expected_choice_count": 3,
            "tool_description": (
                "Use symbolic reasoning with Z3 to solve the ProverQA problem. "
                "Call this when the first-order-logic structure is easier to verify formally than by direct reasoning."
            ),
            "context_description": "The full ProverQA context.",
            "question_description": "The ProverQA statement to classify as true, false, or uncertain.",
        },
    }

    def __init__(self, config: ReasonerConfig, data_loader: DataLoader, answer_extractor: AnswerExtractor):
        super().__init__(config, data_loader, answer_extractor)
        self.dataset_key = self.config.dataset.lower()
        if self.dataset_key not in self.DATASET_SETTINGS:
            supported = ", ".join(sorted(self.DATASET_SETTINGS))
            raise ValueError(f"AdaptiveAgentReasoner currently supports: {supported}.")
        if not callable(getattr(self.api_client, "call_with_messages", None)):
            raise ValueError("AdaptiveAgentReasoner requires an API client with structured message/tool support.")

        self.dataset_settings = self.DATASET_SETTINGS[self.dataset_key]
        prompt_root = os.path.dirname(os.path.abspath(self.config.prompt_path))
        one_step_prompt_path = os.path.join(prompt_root, self.dataset_settings["tool_prompt_dir"])
        if not os.path.exists(one_step_prompt_path):
            raise ValueError(
                f"Adaptive-agent tool prompt path not found: {one_step_prompt_path}. "
                f"Expected a sibling {self.dataset_settings['tool_prompt_dir']} directory next to config.prompt_path."
            )
        self.tool_prompt_handler = PromptHandler(one_step_prompt_path, self.config.dataset)

    def _generate_and_execute_code(self, test_case: dict, solver_name: str, execute_func: Callable, mp_lock: Optional[Any] = None) -> dict:
        """Adaptive-agent does not use the inherited code-generation path."""
        raise NotImplementedError("AdaptiveAgentReasoner uses its own agent/tool loop instead of _generate_and_execute_code.")

    def _build_z3_tool_schema(self) -> List[Dict[str, Any]]:
        """Tool schema exposed to the model."""
        choice_key = self.dataset_settings["choices_key"]
        expected_choice_count = self.dataset_settings["expected_choice_count"]
        return [{
            "type": "function",
            "function": {
                "name": self.dataset_settings["tool_name"],
                "description": self.dataset_settings["tool_description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": self.dataset_settings["context_description"],
                        },
                        "question": {
                            "type": "string",
                            "description": self.dataset_settings["question_description"],
                        },
                        choice_key: {
                            "type": "array",
                            "description": f"The answer choices in order. Expected exactly {expected_choice_count} choices.",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["context", "question", choice_key]
                }
            }
        }]

    @staticmethod
    def _extract_single_answer_index(answer_text: Optional[str]) -> Optional[str]:
        """Extract a single AR-LSAT answer index like [2] from text."""
        if not isinstance(answer_text, str):
            return None
        match = re.search(r"\[\s*([0-4])\s*\]", answer_text)
        if match:
            return f"[{match.group(1)}]"
        normalized = answer_text.strip()
        choice_match = re.search(
            r"\b(?:choice|option)\s+([1-5])\s+(?:is\s+possible|could\s+be|is\s+correct|is\s+the\s+answer)\b",
            normalized,
            re.IGNORECASE,
        )
        if choice_match:
            return f"[{int(choice_match.group(1)) - 1}]"
        return None

    @staticmethod
    def _extract_single_option_letter(answer_text: Optional[str]) -> Optional[str]:
        """Extract a ProverQA answer label like A/B/C from text."""
        if not isinstance(answer_text, str):
            return None

        letter_match = re.search(r"\b(?:the correct option is:\s*)?([ABC])\b", answer_text.strip(), re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

        normalized = answer_text.strip().lower()
        word_map = {
            "true": "A",
            "false": "B",
            "uncertain": "C",
            "unknown": "C",
            "a) true": "A",
            "b) false": "B",
            "c) uncertain": "C",
        }
        return word_map.get(normalized)

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        """Normalize answer text for approximate matching."""
        return re.sub(r"\s+", " ", text.strip().strip("\"'")).lower()

    @classmethod
    def _extract_answer_from_solver_output(cls, solver_output: Optional[str], answers: List[str]) -> Optional[str]:
        """Extract a single AR-LSAT answer index from flexible solver outputs."""
        parsed_answer = cls._extract_single_answer_index(solver_output)
        if parsed_answer:
            return parsed_answer
        if not isinstance(solver_output, str):
            return None

        normalized_output = cls._normalize_answer_text(solver_output)
        normalized_answers = [cls._normalize_answer_text(answer) for answer in answers]

        # Direct exact match to one answer choice.
        exact_matches = [idx for idx, answer in enumerate(normalized_answers) if answer == normalized_output]
        if len(exact_matches) == 1:
            return f"[{exact_matches[0]}]"

        # Single-choice substring match against the whole solver output.
        whole_matches = [idx for idx, answer in enumerate(normalized_answers) if answer and answer in normalized_output]
        if len(whole_matches) == 1:
            return f"[{whole_matches[0]}]"

        # Also allow a short solver output that is a unique substring of one answer choice.
        reverse_matches = [idx for idx, answer in enumerate(normalized_answers) if normalized_output and normalized_output in answer]
        if len(reverse_matches) == 1:
            return f"[{reverse_matches[0]}]"

        positive_markers = (
            "must be true",
            "could be true",
            "could be",
            "is possible",
            "possible",
            "unique",
            "except",
            "correct",
            "answer",
        )
        negative_markers = (
            "cannot be true",
            "not unique",
            "not necessarily true",
            "cannot be",
            "not possible",
            "impossible",
            "incorrect",
        )

        positive_candidates = []
        for raw_line in solver_output.splitlines():
            line = cls._normalize_answer_text(raw_line)
            if not line:
                continue
            if any(marker in line for marker in negative_markers):
                continue
            if not any(marker in line for marker in positive_markers):
                continue
            line_matches = [idx for idx, answer in enumerate(normalized_answers) if answer and answer in line]
            if len(line_matches) == 1:
                positive_candidates.append(line_matches[0])

        unique_candidates = sorted(set(positive_candidates))
        if len(unique_candidates) == 1:
            return f"[{unique_candidates[0]}]"

        return None

    @classmethod
    def _extract_proverqa_answer_from_solver_output(cls, solver_output: Optional[str], options: List[str]) -> Optional[str]:
        """Extract a ProverQA answer label from flexible solver outputs."""
        parsed_letter = cls._extract_single_option_letter(solver_output)
        if parsed_letter:
            return parsed_letter
        if not isinstance(solver_output, str):
            return None

        normalized_output = cls._normalize_answer_text(solver_output)
        normalized_options = [cls._normalize_answer_text(option) for option in options]
        option_to_letter = {idx: chr(ord("A") + idx) for idx in range(len(options))}

        word_map = {
            "true": "A",
            "false": "B",
            "uncertain": "C",
            "unknown": "C",
        }
        if normalized_output in word_map:
            return word_map[normalized_output]

        exact_matches = [idx for idx, option in enumerate(normalized_options) if option == normalized_output]
        if len(exact_matches) == 1:
            return option_to_letter[exact_matches[0]]

        whole_matches = [idx for idx, option in enumerate(normalized_options) if option and option in normalized_output]
        if len(whole_matches) == 1:
            return option_to_letter[whole_matches[0]]

        reverse_matches = [idx for idx, option in enumerate(normalized_options) if normalized_output and normalized_output in option]
        if len(reverse_matches) == 1:
            return option_to_letter[reverse_matches[0]]

        return None

    @staticmethod
    def _assistant_message_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a structured response into a conversation message."""
        message = {
            "role": response.get("role", "assistant"),
            "content": response.get("content") or "",
        }
        tool_calls = response.get("tool_calls") or []
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    @staticmethod
    def _normalize_tool_arguments(arguments: Any) -> Dict[str, Any]:
        """Normalize tool-call arguments into a dictionary."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            return json.loads(arguments)
        raise ValueError("Tool arguments must be a dict or JSON string.")

    def _run_z3_tool(
        self,
        tool_args: Dict[str, Any],
        mp_lock: Optional[Any],
        path_idx: int,
        tool_call_idx: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Execute the dataset-specific Z3 tool and return both log data and tool content."""
        choices_key = self.dataset_settings["choices_key"]
        expected_choice_count = self.dataset_settings["expected_choice_count"]
        choices = tool_args.get(choices_key)
        if isinstance(choices, str):
            choices = json.loads(choices)
        if not isinstance(choices, list) or len(choices) != expected_choice_count:
            raise ValueError(f"{choices_key} must be a list of exactly {expected_choice_count} answer choices.")

        synthetic_test_case = {
            "context": tool_args["context"],
            "question": tool_args["question"],
            choices_key: choices,
        }

        code_prompt = self.tool_prompt_handler.get_direct_prompt(synthetic_test_case)
        code_response = self._call_api(code_prompt)
        code_generation_usage = self._extract_usage(self.api_client)
        code_generation_api_time_only = self._extract_api_time(self.api_client)
        generated_code = CodeCleaner.clean_code(code_response, self.config.dataset)

        final_code, solver_output, is_valid, repair_round_logs, solver_error_type, solver_time_only = self._execute_with_repair(
            synthetic_test_case,
            generated_code,
            self.code_executor.execute_z3_code,
            mp_lock,
            f"adaptive-agent path={path_idx} tool_call={tool_call_idx}",
            prompt_handler=self.tool_prompt_handler,
            repair_client=self.api_client,
        )

        repair_usage_total = self._sum_token_usages([log.get("usage") for log in repair_round_logs])
        repair_api_time_only_total = self._sum_api_times([log.get("api_time_only") for log in repair_round_logs])
        token_usage_total = self._sum_token_usages([code_generation_usage, repair_usage_total])
        api_time_only_total = self._sum_api_times([code_generation_api_time_only, repair_api_time_only_total])
        if self.dataset_key == "ar-lsat":
            parsed_answer = self._extract_answer_from_solver_output(solver_output, choices)
            answer_text = f"The correct option is: {parsed_answer}" if parsed_answer else None
        elif self.dataset_key == "proverqa":
            parsed_answer = self._extract_proverqa_answer_from_solver_output(solver_output, choices)
            answer_text = f"The correct option is: {parsed_answer}" if parsed_answer else None
        else:
            raise ValueError(f"Unsupported adaptive-agent dataset: {self.dataset_key}")

        tool_result_payload = {
            "status": "ok" if is_valid else "error",
            "solver_output": solver_output,
            "parsed_answer": parsed_answer,
            "answer_text": answer_text,
            "solver_error_type": solver_error_type,
        }
        tool_log = {
            "tool_name": self.dataset_settings["tool_name"],
            "tool_call_index": tool_call_idx,
            "path_idx": path_idx,
            "arguments": synthetic_test_case,
            "generated_code": final_code,
            "raw_generated_code": generated_code,
            "code_generation_usage": code_generation_usage,
            "code_generation_api_time_only": code_generation_api_time_only,
            "repair_usage_total": repair_usage_total,
            "repair_api_time_only_total": repair_api_time_only_total,
            "token_usage_total": token_usage_total,
            "api_time_only_total": api_time_only_total,
            "solver_time_only": solver_time_only,
            "repair_round_logs": repair_round_logs,
            "solver_output": solver_output,
            "solver_error_type": solver_error_type,
            "tool_result": tool_result_payload,
        }
        return tool_log, json.dumps(tool_result_payload, ensure_ascii=False)

    def _run_single_agent_path(self, test_case: Dict, path_idx: int, mp_lock: Optional[Any]) -> Dict[str, Any]:
        """Run one adaptive-agent rollout."""
        messages = [{"role": "user", "content": self.prompt_handler.get_agent_prompt(test_case)}]
        tools = self._build_z3_tool_schema()
        force_tool_name = getattr(self.config, "force_tool_name", None)
        tool_choice: Union[str, Dict[str, Any]] = "auto"
        if force_tool_name:
            tool_choice = {
                "type": "function",
                "function": {"name": force_tool_name},
            }
        active_tools: Optional[List[Dict[str, Any]]] = tools
        active_tool_choice: Optional[Union[str, Dict[str, Any]]] = tool_choice

        z3_call_count = 0
        tool_logs: List[Dict[str, Any]] = []
        agent_step_logs: List[Dict[str, Any]] = []
        token_usage_total = self._zero_token_usage()
        api_time_only_total = self._zero_api_time()
        solver_time_only_total = self._zero_solver_time()
        final_response = ""
        last_tool_answer_text = None
        consecutive_tool_error_count = 0
        forced_tool_reminder_count = 0

        for step_idx in range(1, self.MAX_AGENT_STEPS + 1):
            request_kwargs: Dict[str, Any] = {}
            if active_tools is not None:
                request_kwargs["tools"] = active_tools
                if active_tool_choice is not None:
                    request_kwargs["tool_choice"] = active_tool_choice
            response = self._call_api_with_messages(messages, **request_kwargs)
            response_usage = self._extract_usage(self.api_client)
            response_api_time_only = self._extract_api_time(self.api_client)
            token_usage_total = self._add_token_usage(token_usage_total, response_usage)
            api_time_only_total += response_api_time_only

            tool_calls = response.get("tool_calls") or []
            agent_step_logs.append({
                "step_idx": step_idx,
                "response_content": response.get("content"),
                "tool_calls": tool_calls,
                "finish_reason": response.get("finish_reason"),
                "usage": response_usage,
                "api_time_only": response_api_time_only,
                "error": response.get("error"),
            })
            messages.append(self._assistant_message_from_response(response))

            if tool_calls:
                step_had_tool_error = False
                step_had_successful_tool = False
                for tool_call in tool_calls:
                    z3_call_count += 1
                    if z3_call_count > self.MAX_Z3_TOOL_CALLS:
                        limit_payload = {
                            "status": "error",
                            "message": "Maximum z3 tool calls reached. Answer directly using the current information."
                        }
                        step_had_tool_error = True
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "content": json.dumps(limit_payload, ensure_ascii=False),
                        })
                        continue

                    try:
                        tool_args = self._normalize_tool_arguments(tool_call.get("function", {}).get("arguments"))
                        tool_log, tool_result_content = self._run_z3_tool(tool_args, mp_lock, path_idx, z3_call_count)
                    except Exception as exc:
                        tool_log = {
                            "tool_name": tool_call.get("function", {}).get("name"),
                            "tool_call_index": z3_call_count,
                            "path_idx": path_idx,
                            "arguments": tool_call.get("function", {}).get("arguments"),
                            "token_usage_total": self._zero_token_usage(),
                            "api_time_only_total": self._zero_api_time(),
                            "solver_time_only": self._zero_solver_time(),
                            "repair_round_logs": [],
                            "solver_output": str(exc),
                            "solver_error_type": "tool_argument_error",
                            "tool_result": {
                                "status": "error",
                                "message": str(exc),
                                "parsed_answer": None,
                                "answer_text": None,
                            }
                        }
                        tool_result_content = json.dumps(tool_log["tool_result"], ensure_ascii=False)

                    tool_logs.append(tool_log)
                    token_usage_total = self._add_token_usage(token_usage_total, tool_log.get("token_usage_total"))
                    api_time_only_total += self._normalize_api_time(tool_log.get("api_time_only_total"))
                    solver_time_only_total += self._normalize_solver_time(tool_log.get("solver_time_only"))
                    if tool_log.get("tool_result", {}).get("status") == "ok":
                        step_had_successful_tool = True
                    else:
                        step_had_tool_error = True
                    tool_answer_text = tool_log.get("tool_result", {}).get("answer_text")
                    if tool_answer_text:
                        last_tool_answer_text = tool_answer_text
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": tool_result_content,
                    })

                if step_had_successful_tool:
                    consecutive_tool_error_count = 0
                elif step_had_tool_error:
                    consecutive_tool_error_count += 1

                if (
                    consecutive_tool_error_count >= 2
                    or z3_call_count >= self.MAX_Z3_TOOL_CALLS
                ) and active_tools is not None:
                    active_tools = None
                    active_tool_choice = None
                    messages.append({
                        "role": "user",
                        "content": (
                            "The tool has failed repeatedly or the tool-call budget is exhausted. "
                            "Do not call any tools again. Reason directly from the problem and the "
                            "previous tool outputs, then give your best final answer in the required format."
                        ),
                    })
                continue

            if force_tool_name and active_tools is not None and z3_call_count == 0 and forced_tool_reminder_count < 2:
                forced_tool_reminder_count += 1
                messages.append({
                    "role": "user",
                    "content": (
                        f"You must call the tool `{force_tool_name}` now. "
                        "Do not answer directly. Return a tool call instead of a natural-language answer."
                    ),
                })
                continue

            final_response = response.get("content") or ""
            break

        if not final_response:
            if last_tool_answer_text:
                final_response = last_tool_answer_text
            else:
                final_response = "[ERROR: No final answer generated]"

        final_answer_source = "z3" if z3_call_count > 0 else "direct"
        return {
            "path_idx": path_idx,
            "final_response": final_response,
            "z3_called": z3_call_count > 0,
            "z3_call_count": z3_call_count,
            "final_answer_source": final_answer_source,
            "tool_logs": tool_logs,
            "agent_step_logs": agent_step_logs,
            "path_usage_total": token_usage_total,
            "path_api_time_only": api_time_only_total,
            "solver_time_only": solver_time_only_total,
        }

    def reason(self, test_case: Dict, mp_lock: Optional[Any] = None) -> Dict:
        """Run adaptive-agent reasoning for one sample."""
        all_agent_results = []
        for path_idx in range(1, self.config.num_paths + 1):
            print(f"\nRunning adaptive-agent path {path_idx}/{self.config.num_paths}...")
            self.api_client.temperature = self.config.code_temp
            all_agent_results.append(self._run_single_agent_path(test_case, path_idx, mp_lock))
        return {"all_agent_results": all_agent_results}

    def _process_results(self, test_case: Dict, reasoning_result: Dict, case_time: float, unique_id: str = "", timing_data: Optional[Dict] = None) -> None:
        """Save adaptive-agent outputs and tool usage to the results folder."""
        reasoning_folder = os.path.join(self.results_folder, "reasoning")
        code_folder = os.path.join(self.results_folder, "code")
        os.makedirs(reasoning_folder, exist_ok=True)
        os.makedirs(code_folder, exist_ok=True)

        problem_name = test_case['id_string'] if 'id_string' in test_case else test_case['id']
        all_agent_results = reasoning_result.get("all_agent_results", [])
        all_reasoning_outputs = []
        all_solver_outputs = []
        path_token_usage = []
        token_usage_total = self._zero_token_usage()
        api_time_only_total = self._zero_api_time()
        solver_time_only_total = self._zero_solver_time()
        sample_z3_called = False
        sample_z3_call_count = 0
        sample_answer_sources = set()

        for agent_result in all_agent_results:
            path_idx = agent_result["path_idx"]
            final_response = agent_result.get("final_response", "")
            reasoning_filepath = os.path.join(reasoning_folder, f"{problem_name}-{unique_id}-agent{path_idx}.txt")
            with open(reasoning_filepath, "w", encoding="utf-8") as f:
                f.write(final_response)

            for tool_log in agent_result.get("tool_logs", []):
                generated_code = tool_log.get("generated_code")
                if generated_code:
                    code_filepath = os.path.join(
                        code_folder,
                        f"{problem_name}-{unique_id}-agent{path_idx}-z3call{tool_log['tool_call_index']}.py",
                    )
                    with open(code_filepath, "w", encoding="utf-8") as f:
                        f.write(generated_code)

            if final_response == "[ERROR: No final answer generated]":
                is_correct = False
                extracted_answer_index = "[]"
                error_type = "API failure - no final answer generated"
            elif self.dataset_key == "ar-lsat":
                is_correct, extracted_answer_index, error_type = self.answer_extractor.extract_answer(
                    final_response,
                    test_case["label"],
                    "cot",
                )
            elif self.dataset_key == "proverqa":
                is_correct, extracted_answer_index, error_type = self.answer_extractor.extract_answer(
                    final_response,
                    test_case["answer"],
                    "cot",
                )
            else:
                raise ValueError(f"Unsupported adaptive-agent dataset: {self.dataset_key}")

            all_solver_outputs.append(extracted_answer_index)
            all_reasoning_outputs.append({
                "path_idx": path_idx,
                "reasoning_output": final_response,
                "error_type": error_type,
                "success": is_correct,
                "z3_called": agent_result.get("z3_called", False),
                "z3_call_count": agent_result.get("z3_call_count", 0),
                "final_answer_source": agent_result.get("final_answer_source"),
                "tool_logs": agent_result.get("tool_logs", []),
                "agent_step_logs": agent_result.get("agent_step_logs", []),
                "usage": self._normalize_token_usage(agent_result.get("path_usage_total")),
                "api_time_only": self._normalize_api_time(agent_result.get("path_api_time_only")),
                "solver_time_only": self._normalize_solver_time(agent_result.get("solver_time_only")),
            })

            path_usage_total = self._normalize_token_usage(agent_result.get("path_usage_total"))
            path_api_time_only = self._normalize_api_time(agent_result.get("path_api_time_only"))
            path_solver_time_only = self._normalize_solver_time(agent_result.get("solver_time_only"))
            token_usage_total = self._add_token_usage(token_usage_total, path_usage_total)
            api_time_only_total += path_api_time_only
            solver_time_only_total += path_solver_time_only
            path_token_usage.append({
                "path_idx": path_idx,
                "path_usage_total": path_usage_total,
                "path_api_time_only": path_api_time_only,
                "solver_time_only": path_solver_time_only,
                "z3_called": agent_result.get("z3_called", False),
                "z3_call_count": agent_result.get("z3_call_count", 0),
                "final_answer_source": agent_result.get("final_answer_source"),
            })

            sample_z3_called = sample_z3_called or agent_result.get("z3_called", False)
            sample_z3_call_count += int(agent_result.get("z3_call_count", 0))
            sample_answer_sources.add(agent_result.get("final_answer_source"))

        if len(sample_answer_sources) == 1:
            sample_final_answer_source = next(iter(sample_answer_sources))
        elif len(sample_answer_sources) == 0:
            sample_final_answer_source = "none"
        else:
            sample_final_answer_source = "mixed"

        if len(all_reasoning_outputs) == 1:
            sample_reasoning_output = all_reasoning_outputs[0].get("reasoning_output")
            sample_success = all_reasoning_outputs[0].get("success")
            sample_error_type = all_reasoning_outputs[0].get("error_type")
        else:
            sample_reasoning_output = None
            sample_success = None
            sample_error_type = None

        results = {
            "problem": test_case,
            "timing": case_time,
            "reasoning_output": sample_reasoning_output,
            "success": sample_success,
            "error_type": sample_error_type,
            "all_reasoning_outputs": all_reasoning_outputs,
            "all_solver_outputs": all_solver_outputs,
            "path_token_usage": path_token_usage,
            "token_usage_total": token_usage_total,
            "api_time_only_total": api_time_only_total,
            "solver_time_only_total": solver_time_only_total,
            "z3_called": sample_z3_called,
            "z3_call_count": sample_z3_call_count,
            "final_answer_source": sample_final_answer_source,
        }

        summary_filepath = os.path.join(self.summary_folder, f"{problem_name}-{unique_id}.json")
        with open(summary_filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
