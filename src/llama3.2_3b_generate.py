from unsloth import FastLanguageModel
import torch
import time
import os

# Clear any existing CUDA memory
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

def process_tsv_and_generate(input_file, output_dir="queries3b", num_lines=1000):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i in range(num_lines):
            line = f.readline().strip()
            if not line:
                break
                
            # Split line into doc_id and document
            try:
                doc_id, document = line.split('\t', 1)
            except ValueError:
                print(f"Skipping malformed line {i}")
                continue
            
            # Generate queries
            prompt = f"Generate one relevant query for this passage:\n\n{document}. Do not produce anything else."
            
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
                top_k=50,
                do_sample=True,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            
            results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            # Save to individual file for this doc_id
            output_file = os.path.join(output_dir, f"queries_{doc_id}.txt")
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for j, result in enumerate(results, 1):
                    out_f.write(f"Output {j}:\n{result}\n{'-' * 50}\n")
                out_f.write(f"\nExecution time: {time.time() - start_time:.2f} seconds\n")
            
            print(f"Processed document {doc_id} ({i+1}/{num_lines})")
            
            # Optional: Clear GPU memory periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()

# Run the processing
input_file = "/drive_reader/as16386/Datasets/msmarco-2018.tsv"
process_tsv_and_generate(input_file)