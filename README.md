# Qwen Text Classification Inference

This repository contains inference scripts for various text classification tasks using Qwen2.5 models fine-tuned on Vietnamese datasets.

## Setup

1. **Install Dependencies**
   ```bash
   pip install transformers torch tqdm
   ```

2. **Download Checkpoints**
   Download the fine-tuned checkpoints from Hugging Face and place them in the `Txt_CLS/` directory with the following structure:
   ```
   Txt_CLS/
   ├── Qwenv2.5_COLA_results/checkpoint-241/
   ├── Qwenv2.5_MNLI_results/checkpoint-11045/
   ├── Qwenv2.5_MRPC_results/checkpoint-230/
   ├── Qwenv2.5_QNLI_results/checkpoint-2946/
   ├── Qwenv2.5_VSFC_results/checkpoint-358/
   ├── Qwenv2.5_VSMEC_results/checkpoint-348/
   ├── Qwenv2.5_VTOC_results/checkpoint-412/
   ├── Qwenv2.5_WNLI_results/checkpoint-54/
   ├── QQP_classifier/checkpoint-11371/
   └── SST2_classifier/checkpoint-5682/
   ```

## Usage

### Run All Tasks
```bash
python main.py
```

### Run Individual Tasks
```bash
# COLA (Corpus of Linguistic Acceptability)
python scripts/cola_inference.py

# MNLI (Multi-Genre Natural Language Inference)
python scripts/mnli_inference.py

# MRPC (Microsoft Research Paraphrase Corpus)
python scripts/mrpc_inference.py

# QNLI (Question Natural Language Inference)
python scripts/qnli_inference.py

# QQP (Quora Question Pairs)
python scripts/qqp_inference.py

# SST-2 (Stanford Sentiment Treebank)
python scripts/sst2_inference.py

# VSFC (Vietnamese Sentiment Classification)
python scripts/vsfc_inference.py

# VSMEC (Vietnamese Sentiment and Emotion Classification)
python scripts/vsmec_inference.py

# VTOC (Vietnamese Topic Classification)
python scripts/vtoc_inference.py

# WNLI (Winograd Natural Language Inference)
python scripts/wnli_inference.py
```

### Quick Testing
To test with a limited number of examples, modify the `MAX_EXAMPLES` variable in `main.py` or set it in individual scripts:
```python
MAX_EXAMPLES = 20  # Test with 20 examples
```

## Output

### Individual Results
Each script generates a JSON file in the `results/` directory:
- `results/cola_inference_results.json`
- `results/mnli_inference_results.json`
- `results/mrpc_inference_results.json`
- etc.

Format:
```json
{
  "accuracy": 0.85,
  "results": [
    {
      "sentence": "example text",
      "predicted": "label",
      "true_label": "label"
    }
  ]
}
```

### Summary Results
Running `main.py` generates `results/all_inference_summary.json`:
```json
{
  "cola": {"accuracy": 0.85},
  "mnli": {"accuracy": 0.78},
  "mrpc": {"accuracy": 0.92},
  ...
}
```

## Project Structure
```
├── main.py                          # Main script to run all tasks
├── scripts/                         # Individual inference scripts
│   ├── cola_inference.py
│   ├── mnli_inference.py
│   ├── mrpc_inference.py
│   ├── qnli_inference.py
│   ├── qqp_inference.py
│   ├── sst2_inference.py
│   ├── vsfc_inference.py
│   ├── vsmec_inference.py
│   ├── vtoc_inference.py
│   └── wnli_inference.py
├── data/                            # Input data files
├── Txt_CLS/                         # Model checkpoints
└── results/                         # Output results
```

## Supported Tasks

| Task | Description | Labels |
|------|-------------|---------|
| COLA | Corpus of Linguistic Acceptability | acceptable, unacceptable |
| MNLI | Multi-Genre Natural Language Inference | entailment, neutral, contradiction |
| MRPC | Microsoft Research Paraphrase Corpus | equivalent, not_equivalent |
| QNLI | Question Natural Language Inference | entailment, non_entailment |
| QQP | Quora Question Pairs | duplicates, not_duplicates |
| SST-2 | Stanford Sentiment Treebank | positive, negative |
| VSFC | Vietnamese Sentiment Classification | negative, neutral, positive |
| VSMEC | Vietnamese Sentiment and Emotion Classification | anger, disgust, enjoyment, fear, other, sadness, surprise |
| VTOC | Vietnamese Topic Classification | 15 categories (Automobile, Business, Digital, etc.) |
| WNLI | Winograd Natural Language Inference | entailment, non_entailment |

## Notes

- All models use Qwen2.5 architecture with Vietnamese fine-tuning
- Input data should be in JSON format with appropriate fields for each task
- Results include both predictions and accuracy metrics (when true labels are available)
- Scripts support both full dataset evaluation and quick testing with limited examples
