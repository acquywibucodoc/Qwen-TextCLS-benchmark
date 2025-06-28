import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def run_qqp_inference(max_examples=None):
    # Paths
    DATA_PATH = 'data/qqp.json'
    CHECKPOINT_PATH = 'Txt_CLS/QQP_classifier/checkpoint-11371'
    RESULT_PATH = 'results/qqp_inference_results.json'

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
    id2label = {0: 'not duplicates', 1: 'duplicates'}
    label2id = {'not duplicates': 0, 'duplicates': 1}

    def format_prompt(question1, question2):
        return f"<Question1>: {question1} </Question1> <Question2>: {question2} </Question2>"

    results = []
    correct = 0
    has_true_label = 'label' in data[0]

    tqdm_bar = tqdm(data, desc='Inferencing')
    for item in tqdm_bar:
        question1 = item['question1']
        question2 = item['question2']
        prompt = format_prompt(question1, question2)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
        result = {
            'question1': question1,
            'question2': question2,
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
    run_qqp_inference(max_examples=None)  # Change or set to an integer for quick test 