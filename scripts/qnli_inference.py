import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

def run_qnli_inference(max_examples=None):
    # Paths
    DATA_PATH = 'data/qnli.json'
    CHECKPOINT_PATH = 'Txt_CLS/Qwenv2.5_QNLI_results/checkpoint-2946'
    RESULT_PATH = 'results/qnli_inference_results.json'

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

    def format_prompt(question, sentence):
        return f"<Question>: {question} </Question> <Sentence>: {sentence} </Sentence>"

    results = []
    correct = 0
    has_true_label = 'label' in data[0]

    tqdm_bar = tqdm(data, desc='Inferencing')
    for item in tqdm_bar:
        question = item['question']
        sentence = item['sentence']
        prompt = format_prompt(question, sentence)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
        result = {
            'question': question,
            'sentence': sentence,
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
    run_qnli_inference(max_examples=10)  # Change or set to an integer for quick test 