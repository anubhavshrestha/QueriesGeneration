from unsloth import FastLanguageModel
import torch
import time
import os
import argparse
import json
import sys
from functools import partial

# Make all print statements flush immediately
print = partial(print, flush=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input TSV file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for queries')
    parser.add_argument('--start_id', type=int, default=0, help='Document ID to start from')
    return parser.parse_args()

torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=512,
    dtype=None, 
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

def process_tsv_and_generate(input_file, output_dir, start_id):
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    
    queries_file = os.path.join(output_dir, "queries.jsonl")
    
    # Verify the last document ID in existing JSONL file
    if start_id > 0 and os.path.exists(queries_file):
        with open(queries_file, 'r', encoding='utf-8') as f:
            # Get the last line
            last_line = None
            for line in f:
                last_line = line
            if last_line:
                last_doc = json.loads(last_line)
                last_doc_id = int(last_doc['doc_id'])
                if last_doc_id != start_id - 1:
                    raise ValueError(f"Existing JSONL file's last document ID ({last_doc_id}) does not match the expected ID ({start_id - 1}). Please verify the start_id.")
    
    # Single output file for full outputs
    full_output_file = os.path.join(output_dir, "full_outputs.txt")
    
    # Open files in append mode if starting from middle
    file_mode = 'a' if start_id > 0 else 'w'
    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(full_output_file, file_mode, encoding='utf-8', buffering=1) as out_f, \
         open(queries_file, file_mode, encoding='utf-8', buffering=1) as jsonl_f:
        
        # Skip lines until start_id
        while i < start_id:
            f.readline()
            i += 1
        
        while True:
            line = f.readline().strip()
            if not line:
                break
            
            try:
                doc_id, document = line.split('\t', 1)
            except ValueError:
                print(f"Skipping malformed line {i}")
                i += 1
                continue
            
            prompt = f"Generate one relevant and diverse query for this passage:\n\n{document}\n\nDo not produce anything else. Do not produce explanations or anything.\n"
            
            messages = [
                {"role": "user", "content": prompt},
            ]
            
            start_time = time.time()
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=50,
                num_return_sequences=80,
                temperature=0.9, 
                top_k=100,
                top_p=0.95,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            
            results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            # Write full output to text file
            for j, result in enumerate(results, 1):
                out_f.write(f"Output {j}:\n{result}\n{'-' * 50}\n")
                out_f.flush()
                os.fsync(out_f.fileno())  # Force write to disk
            
            out_f.write(f"\nExecution time: {time.time() - start_time:.2f} seconds\n")
            out_f.write("=" * 100 + "\n")  # Separator between documents
            out_f.flush()
            os.fsync(out_f.fileno())
            
            # Extract queries and save to JSONL
            queries = []
            for result in results:
                # Split by 'assistant' and take the last part
                parts = result.split('assistant')
                if len(parts) > 1:
                    # Get all non-empty lines after 'assistant'
                    lines = [line.strip() for line in parts[-1].split('\n') if line.strip()]
                    # Find first line that doesn't contain unwanted phrases
                    valid_query = None
                    for line in lines:
                        if "i cannot" not in line.lower() and "query" not in line.lower():
                            valid_query = line
                            break
                    if valid_query:
                        queries.append(valid_query)
                else:
                    # Fallback to last non-empty line if 'assistant' not found
                    query_lines = result.strip().split('\n')
                    query = next((line for line in reversed(query_lines) if line.strip()), '')
                    queries.append(query)
            
            jsonl_entry = {
                "doc_id": doc_id,
                "queries": queries
            }
            json.dump(jsonl_entry, jsonl_f)
            jsonl_f.write('\n')
            jsonl_f.flush()
            os.fsync(jsonl_f.fileno())  # Force write to disk
            
            print(f"Processed document {doc_id} ({i+1} documents processed)")
            print(f"Current full_outputs position: {out_f.tell()}")
            print(f"Current JSONL position: {jsonl_f.tell()}")
            sys.stdout.flush()
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
            i += 1

if __name__ == "__main__":
    args = parse_args()
    process_tsv_and_generate(args.input_file, args.output_dir, args.start_id)