import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def run_mnli_inference(max_examples=None):
    # Paths
    DATA_PATH = 'data/mnli.json'
    CHECKPOINT_PATH = 'Txt_CLS/Qwenv2.5_MNLI_results/checkpoint-11045'
    RESULT_PATH = 'results/mnli_inference_results.json'

    # Ensure results directory exists
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    # Load data
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if max_examples is not None:
        data = data[:max_examples]

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    model.eval()

    # Label mapping (from config)
    id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    def format_prompt(premise, hypothesis):
        return f"<Premise>: {premise} </Premise> <Hypothesis>: {hypothesis} </Hypothesis>"

    results = []
    correct = 0
    has_true_label = 'label' in data[0]

    tqdm_bar = tqdm(data, desc='Inferencing')
    for item in tqdm_bar:
        premise = item['premise']
        hypothesis = item['hypothesis']
        prompt = format_prompt(premise, hypothesis)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
        result = {
            'premise': premise,
            'hypothesis': hypothesis,
            'predicted': str(pred_id)
        }
        if has_true_label:
            true_label = str(item['label'])
            result['true_label'] = true_label
            if true_label == str(pred_id):
                correct += 1
        results.append(result)

    if has_true_label:
        accuracy = correct / len(data)
        output = {'accuracy': accuracy, 'results': results}
    else:
        output = {'results': results}

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved inference results to {RESULT_PATH}")

if __name__ == '__main__':
    run_mnli_inference(max_examples=None)  # Change or set to None for all 