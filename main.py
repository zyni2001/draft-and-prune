#!/usr/bin/env python3
# import argparse
# from datetime import datetime
from config import ReasonerConfig
from reasoners import CoTReasoner, TwoStepReasoner, DirectReasoner, AdaptiveAgentReasoner
from data_loaders import DataLoader, Sampler, JSON_Dataset
from answer_extractors import (AR_LSAT_AnswerExtractor, 
                               ProofWriter_AnswerExtractor,
                               FOLIO_AnswerExtractor,
                               ProverQA_AnswerExtractor,
                               ProntoQA_AnswerExtractor,
                               LogicalDeduction_AnswerExtractor)
import os
import sys

def main():
    """Main entry point for the script"""

    # Create configuration with command line arguments
    config = ReasonerConfig.from_yaml(sys.argv[1])
    dataset = JSON_Dataset.from_file(config.test_file)

    # Create dataset loader based on format
    if config.dataset.lower() == "ar-lsat":
        answer_extractor = AR_LSAT_AnswerExtractor()
    elif config.dataset.lower() == "proofwriter":
        answer_extractor = ProofWriter_AnswerExtractor()
    elif config.dataset.lower() == "folio":
        answer_extractor = FOLIO_AnswerExtractor()
    elif config.dataset.lower() == "proverqa":
        answer_extractor = ProverQA_AnswerExtractor()
    elif config.dataset.lower() == "prontoqa":
        answer_extractor = ProntoQA_AnswerExtractor()
    elif config.dataset.lower() == "logicaldeduction":
        answer_extractor = LogicalDeduction_AnswerExtractor()
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    
    # Create sampler and dataloader based on desired indices
    sampler = Sampler(desired_indices=config.desired_indices, dataset_size=len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    
    # Create reasoner based on reasoning method
    if config.reasoning_method == "cot":
        reasoner = CoTReasoner(config, dataloader, answer_extractor)
    elif config.reasoning_method == "one-step":
        reasoner = DirectReasoner(config, dataloader, answer_extractor)
    elif config.reasoning_method == "two-step":
        reasoner = TwoStepReasoner(config, dataloader, answer_extractor)
    elif config.reasoning_method == "adaptive-agent":
        reasoner = AdaptiveAgentReasoner(config, dataloader, answer_extractor)
    elif config.reasoning_method == "three-step":
        # reasoner = ThreeStepReasoner(config, data_loader, answer_extractor)
        raise ValueError(f"Unsupported reasoning method: {config.reasoning_method}")
    else:
        raise ValueError(f"Unsupported reasoning method: {config.reasoning_method}") 
        # raise ValueError(f"Unsupported reasoning method: {config.reasoning_method}")

    # save the config to a yaml file in the results folder
    reasoner.config.save_to_yaml(os.path.join(reasoner.results_folder, "config.yaml"))
    
    # Copy the prompts folder to preserve exact prompts used
    reasoner.save_prompts_folder()

    # Run all tests
    # limit_msg = f" with limit {args.limit}" if args.limit else ""
    # print(f"Running all tests from {args.dataset}{limit_msg} using {args.reasoning_method} reasoning")
    reasoner.run_all_tests_parallel(config.num_processes)
    # reasoner.run_all_tests()


if __name__ == "__main__":
    main() 
