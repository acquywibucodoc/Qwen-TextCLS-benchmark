import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def run_wnli_inference(max_examples=None):
    # Paths
    DATA_PATH = 'data/wnli.json'
    CHECKPOINT_PATH = 'Txt_CLS/Qwenv2.5_WNLI_results/checkpoint-54'
    RESULT_PATH = 'results/wnli_inference_results.json'

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
    id2label = {0: 'entailment', 1: 'non_entailment'}
    label2id = {'entailment': 0, 'non_entailment': 1}

    def format_prompt(sentence1, sentence2):
        return f"<Sentence1>: {sentence1} </Sentence1> <Sentence2>: {sentence2} </Sentence2>"

    results = []
    correct = 0
    has_true_label = 'label' in data[0]

    tqdm_bar = tqdm(data, desc='Inferencing')
    for item in tqdm_bar:
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        prompt = format_prompt(sentence1, sentence2)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = id2label[pred_id]
        result = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'predicted': pred_label
        }
        if has_true_label:
            true_label_idx = item['label']
            if isinstance(true_label_idx, str) and true_label_idx.isdigit():
                true_label_idx = int(true_label_idx)
            elif isinstance(true_label_idx, str):
                true_label_idx = label2id.get(true_label_idx, true_label_idx)
            true_label = id2label[true_label_idx] if isinstance(true_label_idx, int) else true_label_idx
            result['true_label'] = true_label
            if pred_label == true_label:
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
    run_wnli_inference(max_examples=50)  # Change or set to an integer for quick test 