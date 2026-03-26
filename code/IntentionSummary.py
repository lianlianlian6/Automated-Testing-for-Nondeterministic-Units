import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from peft import PeftModel

BASE_MODEL = 'your_path/your-model'
ADAPTER_PATH = 'your_path'
INPUT_PATH = './program.json'
OUTPUT_PATH = INPUT_PATH

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, local_files_only=True)
model = model.merge_and_unload()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_intention(code, max_new_tokens=128):
    prompt = f"Please summarize the intent of the given code as concisely as possible: {code}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][input_ids.shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded.strip()

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"[Error] Input file does not exist: {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in tqdm(data, desc="Generating function intentions"):
        code = entry.get("test_program", "")
        intention = generate_intention(code)
        entry["intention"] = intention

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[✓] All function intentions generated and saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()