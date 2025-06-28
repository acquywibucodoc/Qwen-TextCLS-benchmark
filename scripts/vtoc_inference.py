import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def run_vtoc_inference(max_examples=None):
    # Paths
    DATA_PATH = 'data/vtoc.json'
    CHECKPOINT_PATH = 'Txt_CLS/Qwenv2.5_VTOC_results/checkpoint-412'
    RESULT_PATH = 'results/vtoc_inference_results.json'

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
    id2label = {
        0: 'Automobile', 1: 'Business', 2: 'Digital', 3: 'Education', 4: 'Entertainment',
        5: 'Health', 6: 'Law', 7: 'Life', 8: 'News', 9: 'Perspective', 10: 'Relax',
        11: 'Science', 12: 'Sports', 13: 'Travel', 14: 'World'
    }
    label2id = {v: k for k, v in id2label.items()}

    def format_prompt(sentence):
        return f"<Sentence>: {sentence} </Sentence>"

    results = []
    correct = 0
    has_true_label = 'label' in data[0]

    tqdm_bar = tqdm(data, desc='Inferencing')
    for item in tqdm_bar:
        sentence = item['sentence']
        prompt = format_prompt(sentence)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = id2label[pred_id]
        result = {
            'sentence': sentence,
            'predicted': pred_label
        }
        if has_true_label:
            # Accept both int and str for label in data
            true_label_idx = item['label']
            if isinstance(true_label_idx, str) and true_label_idx.isdigit():
                true_label_idx = int(true_label_idx)
            elif isinstance(true_label_idx, str):
                # If label is already a string name
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
    run_vtoc_inference(max_examples=50)  # Change or set to an integer for quick test 