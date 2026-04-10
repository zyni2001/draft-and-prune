#!/usr/bin/env python3
import re
from typing import Tuple, Optional, List, Dict, Union
from abc import ABC, abstractmethod
from collections import Counter

class AnswerExtractor(ABC):
    """Base class for answer extractors"""
    
    @abstractmethod
    def extract_answer(self, response_text: str, label: str) -> Tuple[bool, str]:
        """Extract the answer from the response text"""
        pass
    

class AR_LSAT_AnswerExtractor(AnswerExtractor):
    """Answer extractor specifically tailored for AR-LSAT dataset format"""
    
    def extract_answer(self, response_text: str, label: str, reasoning_method: str = "cot") -> Tuple[bool, str, Optional[str]]:
        """
        Extract answer from response text and check if it matches the correct label.
        Accepts either "The correct option is: [i]" or "The correct option is: i" where i is 0-4.
        
        Args:
            response_text: The model's response text containing reasoning and answer
            label: The correct answer option index (0, 1, 2, 3, 4) or letter (A, B, C, D, E)
            answers: List of answer choices (optional, for content matching)
            reasoning_method: The reasoning method used (default: "cot")
        
        Returns:
            Tuple[bool, str, Optional[str]]: (is_correct, extracted_answer_index, error_type)
        """
        if reasoning_method == "cot":
            # Convert label to index if it's a letter
            if isinstance(label, int) or label.isdigit():
                label_idx = int(label)
            else:
                return False, '[]', 'invalid label format'

            # Accept either bracketed or plain numeric answers.
            pattern = r"The correct option is:\s*(?:\[([0-4])\]|([0-4]))"
            match = re.search(pattern, response_text, re.IGNORECASE)
            
            if match:
                extracted_idx = int(match.group(1) or match.group(2))
                is_correct = extracted_idx == label_idx
                if is_correct:
                    return True, f"[{extracted_idx}]", None
                else:
                    return False, f"[{extracted_idx}]", 'semantic error'
            else:
                # No match found
                return False, '[]', 'answer not found in choices'
        elif reasoning_method == "one-step" or reasoning_method == "two-step" or reasoning_method == "three-step":
            raise ValueError(f"Unsupported reasoning method: {reasoning_method}")


class ProofWriter_AnswerExtractor(AnswerExtractor):
    """Answer extractor specifically tailored for ProofWriter dataset format"""
    
    def extract_answer(self, response_text: Union[str, Optional[bool]], label: str, reasoning_method: str = "cot") -> Tuple[bool, str, Optional[str]]:
        """
        Extract answer from response text and check if it matches the correct label.
        
        Args:
            response_text: The model's response text containing reasoning and answer
            label: The correct answer option letter (A, B, C)
            reasoning_method: The reasoning method used (default: "cot")
        
        Returns:
            Tuple[bool, str, Optional[str]]: (is_correct, extracted_answer_index, error_type)
        """
        if reasoning_method == "cot":
            keywords = ['A', 'B', 'C', 'True', 'False', 'Unknown']
            positions = {word: response_text.rfind(word) for word in keywords}
            
            valid_positions = {k: v for k, v in positions.items() if v != -1}
            
            if not valid_positions:
                return False, '[]', 'answer not found in choices'
            
            target = max(valid_positions.items(), key=lambda x: x[1])[0]
            if target == 'A' or target == 'True':
                extracted_answer = 'A'
            elif target == 'B' or target == 'False':
                extracted_answer = 'B'
            elif target == 'C' or target == 'Unknown':
                extracted_answer = 'C'
            else:
                return False, '[]', 'semantic error'
            
            is_correct = extracted_answer == label
            if is_correct:
                return True, f"[{extracted_answer}]", None
            else:
                return False, f"[{extracted_answer}]", 'semantic error'
        elif reasoning_method == "one-step" or reasoning_method == "two-step" or reasoning_method == "three-step":
            raise ValueError(f"Unsupported reasoning method: {reasoning_method}")


class FOLIO_AnswerExtractor(AnswerExtractor):
    """Answer extractor specifically tailored for FOLIO dataset format"""
    
    def extract_answer(self, response_text: Union[str, Optional[bool]], label: str, reasoning_method: str = "cot") -> Tuple[bool, str, Optional[str]]:
        """
        Extract answer from response text and check if it matches the correct label.
        
        Args:
            response_text: The model's response text containing reasoning and answer
            label: The correct answer option letter (A, B, C)
            reasoning_method: The reasoning method used (default: "cot")
        
        Returns:
            Tuple[bool, str, Optional[str]]: (is_correct, extracted_answer_index, error_type)
        """
        if reasoning_method == "cot":
            keywords = ['A', 'B', 'C', 'True', 'False', 'Uncertain']
            positions = {word: response_text.rfind(word) for word in keywords}
            
            valid_positions = {k: v for k, v in positions.items() if v != -1}
            
            if not valid_positions:
                return False, '[]', 'answer not found in choices'
            
            target = max(valid_positions.items(), key=lambda x: x[1])[0]
            if target == 'A' or target == 'True':
                extracted_answer = 'A'
            elif target == 'B' or target == 'False':
                extracted_answer = 'B'
            elif target == 'C' or target == 'Uncertain':
                extracted_answer = 'C'
            else:
                return False, '[]', 'semantic error'
            
            is_correct = extracted_answer == label
            if is_correct:
                return True, f"[{extracted_answer}]", None
            else:
                return False, f"[{extracted_answer}]", 'semantic error'
        elif reasoning_method == "one-step" or reasoning_method == "two-step" or reasoning_method == "three-step":
            raise ValueError(f"Unsupported reasoning method: {reasoning_method}")


class ProverQA_AnswerExtractor(AnswerExtractor):
    """Answer extractor specifically tailored for ProverQA dataset format"""

    def extract_answer(self, response_text: Union[str, Optional[bool]], label: str, reasoning_method: str = "cot") -> Tuple[bool, str, Optional[str]]:
        """
        Extract answer from response text and check if it matches the correct label.

        Args:
            response_text: The model's response text containing reasoning and answer
            label: The correct answer option letter (A, B, C)
            reasoning_method: The reasoning method used (default: "cot")

        Returns:
            Tuple[bool, str, Optional[str]]: (is_correct, extracted_answer_index, error_type)
        """
        if reasoning_method == "cot":
            keywords = ['A', 'B', 'C', 'True', 'False', 'Uncertain', 'Unknown']
            positions = {word: response_text.rfind(word) for word in keywords}

            valid_positions = {k: v for k, v in positions.items() if v != -1}

            if not valid_positions:
                return False, '[]', 'answer not found in choices'

            target = max(valid_positions.items(), key=lambda x: x[1])[0]
            if target == 'A' or target == 'True':
                extracted_answer = 'A'
            elif target == 'B' or target == 'False':
                extracted_answer = 'B'
            elif target in ('C', 'Uncertain', 'Unknown'):
                extracted_answer = 'C'
            else:
                return False, '[]', 'semantic error'

            is_correct = extracted_answer == label
            if is_correct:
                return True, f"[{extracted_answer}]", None
            else:
                return False, f"[{extracted_answer}]", 'semantic error'
        elif reasoning_method == "one-step" or reasoning_method == "two-step" or reasoning_method == "three-step":
            raise ValueError(f"Unsupported reasoning method: {reasoning_method}")


class ProntoQA_AnswerExtractor(AnswerExtractor):
    """Answer extractor specifically tailored for ProntoQA dataset format"""
    
    def extract_answer(self, response_text: Union[str, Optional[bool]], label: str, reasoning_method: str = "cot") -> Tuple[bool, str, Optional[str]]:
        """
        Extract answer from response text and check if it matches the correct label.
        
        Args:
            response_text: The model's response text containing reasoning and answer
            label: The correct answer option letter (A, B)
            reasoning_method: The reasoning method used (default: "cot")
        
        Returns:
            Tuple[bool, str, Optional[str]]: (is_correct, extracted_answer_index, error_type)
        """
        if reasoning_method == "cot":
            keywords = ['A', 'B', 'True', 'False']
            positions = {word: response_text.rfind(word) for word in keywords}
            
            valid_positions = {k: v for k, v in positions.items() if v != -1}
            
            if not valid_positions:
                return False, '[]', 'answer not found in choices'
            
            target = max(valid_positions.items(), key=lambda x: x[1])[0]
            if target == 'A' or target == 'True':
                extracted_answer = 'A'
            elif target == 'B' or target == 'False':
                extracted_answer = 'B'
            else:
                return False, '[]', 'semantic error'
            
            is_correct = extracted_answer == label
            if is_correct:
                return True, f"[{extracted_answer}]", None
            else:
                return False, f"[{extracted_answer}]", 'semantic error'
        elif reasoning_method == "one-step" or reasoning_method == "two-step" or reasoning_method == "three-step":
            raise ValueError(f"Unsupported reasoning method: {reasoning_method}")
    
    
class LogicalDeduction_AnswerExtractor(AnswerExtractor):
    """Answer extractor specifically tailored for LogicalDeduction dataset format"""
    
    def extract_answer(self, response_text: str, label: str, reasoning_method: str = "cot") -> Tuple[bool, str, Optional[str]]:
        """
        Extract answer from response text and check if it matches the correct label.
        
        Args:
            response_text: The model's response text containing reasoning and answer
            label: The correct answer option letter (A, B, C, D, E, F, G)
            reasoning_method: The reasoning method used (default: "cot")
        
        Returns:
            Tuple[bool, str, Optional[str]]: (is_correct, extracted_answer_index, error_type)
        """
        if reasoning_method == "cot":
            keywords = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)']
            positions = {word: response_text.rfind(word) for word in keywords}
            
            valid_positions = {k: v for k, v in positions.items() if v != -1}
            
            if not valid_positions:
                return False, '[]', 'answer not found in choices'
            
            target = max(valid_positions.items(), key=lambda x: x[1])[0]
            if target == 'A)':
                extracted_answer = 'A'
            elif target == 'B)':
                extracted_answer = 'B'
            elif target == 'C)':
                extracted_answer = 'C'
            elif target == 'D)':
                extracted_answer = 'D'
            elif target == 'E)':
                extracted_answer = 'E'
            elif target == 'F)':
                extracted_answer = 'F'
            elif target == 'G)':
                extracted_answer = 'G'
            else:
                return False, '[]', 'semantic error'
            
            is_correct = extracted_answer == label
            if is_correct:
                return True, f"[{extracted_answer}]", None
            else:
                return False, f"[{extracted_answer}]", 'semantic error'
        elif reasoning_method == "one-step" or reasoning_method == "two-step" or reasoning_method == "three-step":
            raise ValueError(f"Unsupported reasoning method: {reasoning_method}")
