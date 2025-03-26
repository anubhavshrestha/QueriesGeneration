#!/usr/bin/env python3
import argparse
import os
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama model for document-to-query generation")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                      help="Model to fine-tune (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to training data file (.tsv/.jsonl)")
    parser.add_argument("--output_dir", type=str, default="doc2query-llama-3.2",
                      help="Output directory for the fine-tuned model")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=32,
                      help="Rank of LoRA update matrices (default: 32)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                      help="LoRA alpha scaling parameter (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                      help="LoRA dropout probability (default: 0.05)")
    
    # Training parameters
    parser.add_argument("--per_device_batch_size", type=int, default=4,  # REDUCED FROM 8
                      help="Batch size per device (default: 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,  # INCREASED FROM 4
                      help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--num_epochs", type=int, default=1,
                      help="Number of training epochs (default: 1)")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4,
                      help="Learning rate (default: 1.5e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                      help="Weight decay (default: 0.001)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Max gradient norm (default: 1.0)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                      help="Warmup ratio (default: 0.03)")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                      choices=["linear", "cosine", "constant"], 
                      help="LR scheduler type (default: cosine)")
    parser.add_argument("--max_seq_length", type=int, default=512,  # REDUCED FROM 512
                      help="Maximum sequence length (default: 512)")
    
    # Other parameters
    parser.add_argument("--use_4bit", action="store_true", default=True,
                      help="Use 4-bit quantization")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                      help="Use bfloat16 precision if available")
    parser.add_argument("--save_steps", type=int, default=500,
                      help="Save checkpoint every X steps (default: 500)")
    parser.add_argument("--logging_steps", type=int, default=50,
                      help="Log every X steps (default: 50)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print GPU info
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Prepare quantization config
    compute_dtype = torch.bfloat16 if args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )
    
    # IMPORTANT: Explicitly set attention implementation (fixes position issue)
    model.config._attn_implementation = "eager"
    model.config.use_cache = False
    
    # Load and prepare tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load dataset
    logger.info(f"Loading training data from: {args.data_path}")
    if args.data_path.endswith('.jsonl'):
        dataset = load_dataset("json", data_files=args.data_path, split="train")
    else:
        dataset = load_dataset("csv", data_files=args.data_path, delimiter="\t", 
                              column_names=["document", "query"], split="train")
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Sample data
    if "document" in dataset.column_names and "query" in dataset.column_names:
        sample_doc = dataset[0]["document"]
        sample_query = dataset[0]["query"]
        logger.info(f"Document: {sample_doc[:100]}...")
        logger.info(f"Query: {sample_query}")
    
    # Format dataset
    def tokenize_function(examples):
        # Create formatted prompts
        prompts = [f"[INST] Generate a search query for this document: {doc} [/INST] {query}" 
                  for doc, query in zip(examples["document"], examples["query"])]
        
        # Ensure proper truncation to prevent indexing errors
        tokenized = tokenizer(
            prompts,
            padding="max_length",  # Always pad to max_length for consistent positions
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors=None
        )
        
        # Create labels (same as input_ids for causal LM)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        
        # Mask the instruction part for loss calculation
        for i, prompt in enumerate(prompts):
            instruction_end = prompt.find("[/INST]") + len("[/INST]")
            instruction_tokens = tokenizer(prompt[:instruction_end], add_special_tokens=False)["input_ids"]
            # Set instruction part labels to -100 to ignore in loss
            for j in range(min(len(instruction_tokens), len(tokenized["labels"][i]))):
                tokenized["labels"][i][j] = -100
        
        return tokenized
    
    # Process smaller batches to avoid memory issues
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,  # Smaller batch size for processing
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # Configure LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    logger.info(f"Configuring LoRA with rank={args.lora_r}, alpha={args.lora_alpha}")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Configure training arguments
    bf16 = args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        group_by_length=False,  # DISABLED - can cause position issues
        report_to="none",
        dataloader_num_workers=0,  # Prevent multiprocessing issues
        dataloader_pin_memory=False,
    )
    
    # Use standard Trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt"
        ),
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
