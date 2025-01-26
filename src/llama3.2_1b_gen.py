from unsloth import FastLanguageModel
import torch
import time
import os
import argparse

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_file', type=str, required=True, help='Input TSV file path')
   parser.add_argument('--output_dir', type=str, required=True, help='Output directory for queries')
   return parser.parse_args()

torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
   model_name="unsloth/Llama-3.2-1B-Instruct",
   max_seq_length=512,
   dtype=None, 
   load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

def process_tsv_and_generate(input_file, output_dir):
   os.makedirs(output_dir, exist_ok=True)
   i = 0
   
   with open(input_file, 'r', encoding='utf-8') as f:
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
           
           prompt = f"Generate one relevant and diverse query for this passage:\n\n{document}\n\nDo not produce anything else. Do not produce explanations or anything."
           
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
               do_sample=True,
               use_cache=True,
               pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
           )
           
           results = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
           
           output_file = os.path.join(output_dir, f"queries_{doc_id}.txt")
           with open(output_file, 'w', encoding='utf-8') as out_f:
               for j, result in enumerate(results, 1):
                   out_f.write(f"Output {j}:\n{result}\n{'-' * 50}\n")
               out_f.write(f"\nExecution time: {time.time() - start_time:.2f} seconds\n")
           
           print(f"Processed document {doc_id} ({i+1} documents processed)")
           
           if i % 10 == 0:
               torch.cuda.empty_cache()
           i += 1

if __name__ == "__main__":
   args = parse_args()
   process_tsv_and_generate(args.input_file, args.output_dir)