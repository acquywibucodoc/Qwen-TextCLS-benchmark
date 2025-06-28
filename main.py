import json
import os
from scripts.cola_inference import run_cola_inference
from scripts.mnli_inference import run_mnli_inference
from scripts.mrpc_inference import run_mrpc_inference
from scripts.qnli_inference import run_qnli_inference
from scripts.qqp_inference import run_qqp_inference
from scripts.sst2_inference import run_sst2_inference
from scripts.vsfc_inference import run_vsfc_inference
from scripts.vsmec_inference import run_vsmec_inference
from scripts.vtoc_inference import run_vtoc_inference
from scripts.wnli_inference import run_wnli_inference

MAX_EXAMPLES = None

def get_accuracy_from_result(result_path):
    if not os.path.exists(result_path):
        return None
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('accuracy', None)

def main():
    # Run all inference scripts (set max_examples=None for full, or a number for quick test)
    run_cola_inference(max_examples=MAX_EXAMPLES)
    run_mnli_inference(max_examples=MAX_EXAMPLES)
    run_mrpc_inference(max_examples=MAX_EXAMPLES)
    run_qnli_inference(max_examples=MAX_EXAMPLES)
    run_qqp_inference(max_examples=MAX_EXAMPLES)
    run_sst2_inference(max_examples=MAX_EXAMPLES)
    run_vsfc_inference(max_examples=MAX_EXAMPLES)
    run_vsmec_inference(max_examples=MAX_EXAMPLES)
    run_vtoc_inference(max_examples=MAX_EXAMPLES)
    run_wnli_inference(max_examples=MAX_EXAMPLES)

    # Collect results
    summary = {}
    result_files = {
        'mnli': 'results/mnli_inference_results.json',
        'qnli': 'results/qnli_inference_results.json',
        'wnli': 'results/wnli_inference_results.json',
        'sst2': 'results/sst2_inference_results.json',
        'vsfc': 'results/vsfc_inference_results.json',
        'vsmec': 'results/vsmec_inference_results.json',
        'mrpc': 'results/mrpc_inference_results.json',
        'qqp': 'results/qqp_inference_results.json',
        'cola': 'results/cola_inference_results.json',
        'vtoc': 'results/vtoc_inference_results.json',
    }
    for task, path in result_files.items():
        acc = get_accuracy_from_result(path)
        summary[task] = {'accuracy': acc}

    # Save summary
    summary_path = 'results/all_inference_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")

if __name__ == '__main__':
    main() 