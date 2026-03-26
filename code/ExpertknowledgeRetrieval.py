import os
import re
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

BASE_MODEL = 'your_path/your_model'
ADAPTER_PATH = 'your_path'
JSON_PATH = './program.json'
TOP_K = 4  

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, local_files_only=True)
model = model.merge_and_unload()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def split_sentences(text):
    return [s.strip() for s in re.split(r'[。.!?]', text) if len(s.strip()) > 5]

def extract_pdf_sentences_from_folder(folder_path):
    all_sentences = []
    if not os.path.exists(folder_path):
        print(f"[Error] Folder not found: {folder_path}")
        return []

    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[Warning] No PDF files found in folder: {folder_path}")
        return []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_sentences.extend(split_sentences(text))
        except Exception as e:
            print(f"[Warning] Failed to read {pdf_path}: {e}")
    return all_sentences

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        embeds = model.get_input_embeddings()(inputs["input_ids"])
        return embeds.mean(dim=1).squeeze().cpu().numpy()

def build_faiss_index(sentences):
    embeddings = [get_embedding(s) for s in tqdm(sentences, desc="Embedding PDF sentences")]
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def enrich_program(folder_path):
    sentences = extract_pdf_sentences_from_folder(folder_path)
    if len(sentences) == 0:
        print("[Error] No valid sentences extracted from PDF files.")
        return

    index, _ = build_faiss_index(sentences)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in tqdm(data, desc="Querying intentions"):
        intention = entry.get("intention", "").strip()
        if not intention:
            entry["text"] = []
            continue

        query_vec = get_embedding(intention).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_vec, TOP_K)

        filtered = []
        for i in indices[0]:
            sent = sentences[i]
            if len(sent.strip()) < 10:
                continue
            if re.fullmatch(r"[\d\s%\.:\-+/,，()（）]+", sent):
                continue
            filtered.append(sent)

        entry["text"] = filtered[:TOP_K]

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Supplementary text added to each test_program in {JSON_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enrich code information based on PDF folder content")
    parser.add_argument("--folder_path", help="Path to a folder containing PDF files")
    args = parser.parse_args()

    enrich_program(args.folder_path)

