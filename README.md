# Draft-and-Prune

A neuro-symbolic tool that combines LLMs with symbolic solvers for logical reasoning using multi-path ensemble with sketched planning, code generation, and path pruning.

Paper: [Draft-and-Prune: Improving the Reliability of Auto-formalization for Logical Reasoning](https://arxiv.org/abs/2603.17233)

**Related: [Agentified Assessment of Logical Reasoning Agents](https://arxiv.org/abs/2603.02788):** an **assessor agent** runs the benchmark loop (tasks, budgets, parsing, failure logs) so evaluation stays **reproducible** when symbolic runs fail; the system under test only needs a standard **A2A** interface.

**Cleaned benchmarks we released:** We also **fixed and republished** two widely used logical-reasoning benchmark — **[FOLIO](https://huggingface.co/datasets/yfxiao/folio-refined)** and **[AR-LSAT](https://huggingface.co/datasets/anonymous-ar-lsat/ar-lsat-fixed-229)**. 

## Quick Start

```bash
git clone https://github.com/zyni2001/draft-and-prune.git
cd draft-and-prune
pip install -r requirements.txt

# Run experiment with a provider-specific template
python main.py configs/azure-openai/config_azure_openai.yaml
```

The old root-level `config_gpt.yaml` was an Azure OpenAI template. Provider-specific templates now live under `configs/`.

## Project Structure

```
├── main.py                     # Main entry point
├── config.py                   # Configuration parsing
├── reasoners.py                # Reasoning logic (CoT, two-step, one-step)
├── answer_extractors.py        # Dataset-specific answer extraction for CoT runs
├── call_api.py                 # API clients (GPT-4, Gemini)
├── data_loaders.py             # Dataset loading
├── path_level_analysis.py      # Path-level analysis
├── analysis_pruning_and_ensemble_simulation.py  # Majority voting simulation
│
├── configs/
│   ├── azure-openai/           # Azure OpenAI templates
│   ├── openai-compatible/      # OpenAI-compatible templates
│   ├── gemini/                 # Gemini templates
│   └── anthropic/              # Anthropic templates
│
├── prompts-all-3-shot-aligned/ # Prompt templates
│   └── {Dataset}-prompts-{method}/  # CoT, one-step, two-step-partition
│
├── data/                       # Datasets (AR-LSAT, ProofWriter, ProntoQA, LogicalDeduction)
└── results/                    # Experiment outputs
```

## Analysis Workflow

```
Step 1: Run Experiment
  python main.py configs/azure-openai/config_azure_openai.yaml
  → results/{experiment}/summary/*.json

Step 2: Path-Level Analysis
  python path_level_analysis.py <results_dir> --expected-paths 20
  → path_level_analysis.csv

Step 3: Majority Voting Simulation
  python analysis_pruning_and_ensemble_simulation.py path_level_analysis.csv --pruning-mode both
  → simulation results with accuracy, exec_rate, exec_acc, hit_rate for k=1..20
```

---

### Path-Level Analysis

```bash
# Basic usage
python path_level_analysis.py <results_dir> --expected-paths 20

# With CoT backup
python path_level_analysis.py <results_dir> --expected-paths 20 \
  --cot-summary path/to/cot_summary.txt --cot-backup
```

**Options:**

- `--expected-paths N`: Paths per sample (default: 1)
- `--output-format`: csv, xlsx, or both (default: csv)
- `--cot-summary`: Add `cot_correctness` column from CoT results
- `--cot-backup`: Use CoT as backup for syntax errors

**Output:** `path_level_analysis.csv` with path correctness, pruning results, and statistics.

---

### Majority Voting Simulation

```bash
# Without pruning
python analysis_pruning_and_ensemble_simulation.py input.csv --pruning-mode no_pruning

# With pruning + CoT fallback
python analysis_pruning_and_ensemble_simulation.py input.csv --pruning-mode both --cot-backup
```

**Options:**

- `--pruning-mode`: `no_pruning`, `existence`, `uniqueness`, `both`, or `all`
- `--cot-backup`: Use CoT result when no valid symbolic output
- `--max-paths`: Maximum k value (default: 20)
- `--num-simulations`: Iterations per k (default: 10)

**Output Metrics:**


| Metric      | Formula                     | Description                   |
| ----------- | --------------------------- | ----------------------------- |
| `accuracy`  | correct / total             | Overall accuracy              |
| `exec_rate` | exec_samples / total        | Fraction with valid outputs   |
| `exec_acc`  | exec_correct / exec_samples | Accuracy among executable     |
| `hit_rate`  | hit_samples / total         | Fraction with ≥1 correct path |


---

## Supported Datasets


| Dataset                                | Source                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------- |
| AR-LSAT (original)                     | [HuggingFace](https://huggingface.co/datasets/tasksource/lsat-ar)                  |
| AR-LSAT — **cleaned** 229-sample split | [HuggingFace](https://huggingface.co/datasets/anonymous-ar-lsat/ar-lsat-fixed-229) |
| FOLIO — **cleaned** (FOLIO-Refined)    | [HuggingFace](https://huggingface.co/datasets/yfxiao/folio-refined)                |
| ProofWriter                            | [HuggingFace](https://huggingface.co/datasets/tasksource/proofwriter)              |
| ProntoQA                               | [HuggingFace](https://huggingface.co/datasets/renma/ProntoQA)                      |
| LogicalDeduction                       | [HuggingFace](https://huggingface.co/datasets/maveriq/bigbenchhard)                |


## Config Templates

- Azure OpenAI: `configs/azure-openai/config_azure_openai.yaml`
- OpenAI-compatible: `configs/openai-compatible/config_openai_compatible.yaml`
- Gemini: `configs/gemini/config_gemini.yaml`
- Anthropic: `configs/anthropic/config_anthropic.yaml`

## Citation

If you use this repository, please cite:

```bibtex
@article{ni2026draft,
  title={Draft-and-Prune: Improving the Reliability of Auto-formalization for Logical Reasoning},
  author={Ni, Zhiyu and Liang, Zheng and Song, Liangcheng and Cao, Chenrui and Zhang, Xian and Sangiovanni-Vincentelli, Alberto and Nuzzo, Pierluigi},
  journal={arXiv preprint arXiv:2603.17233},
  year={2026}
}
```

If you use the FOLIO-Refined dataset or the agentified assessment framework, please also cite:

```bibtex
@article{ni2026agentified,
  title={Agentified Assessment of Logical Reasoning Agents},
  author={Ni, Zhiyu and Xiao, Yifeng and Liang, Zheng},
  journal={arXiv preprint arXiv:2603.02788},
  year={2026}
}
```

## License

[Include your license information here]
