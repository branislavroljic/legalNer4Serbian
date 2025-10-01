# MLM Pretraining Configuration (add to notebook)
MLM_CONFIG = {
    "num_epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "warmup_steps": 500,
    "save_steps": 1000,
    "eval_steps": 500,
    "mlm_probability": 0.15,
    "max_length": 512,
    "stride": 256
}

# MLM Functions (add to notebook)
def load_mlm_documents(judgments_dir: str) -> List[str]:
    """Load documents for MLM pretraining"""
    documents = []
    
    if not os.path.exists(judgments_dir):
        print(f"❌ Judgments directory not found: {judgments_dir}")
        return documents
    
    txt_files = [f for f in os.listdir(judgments_dir) if f.endswith('.txt')]
    
    for filename in tqdm(txt_files, desc="Loading MLM documents"):
        file_path = os.path.join(judgments_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
        except Exception as e:
            print(f"⚠️  Error reading {filename}: {e}")
    
    print(f"📚 Loaded {len(documents)} documents for MLM pretraining")
    return documents


def preprocess_mlm_data(documents: List[str], tokenizer, max_length: int = 512, stride: int = 256):
    """Preprocess documents for MLM training"""
    from datasets import Dataset
    
    # Split long documents into chunks
    chunks = []
    for doc in tqdm(documents, desc="Processing MLM documents"):
        # Tokenize document
        tokens = tokenizer.tokenize(doc)
        
        # Split into overlapping chunks
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + max_length - 2]  # -2 for [CLS] and [SEP]
            if len(chunk_tokens) > 10:  # Only keep chunks with sufficient content
                chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
    
    print(f"📄 Created {len(chunks)} text chunks for MLM training")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": chunks})
    
    # Tokenize for MLM
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    return tokenized_dataset


def perform_mlm_pretraining(model, tokenizer, train_dataset, output_dir: str, mlm_config: dict):
    """Perform MLM pretraining on legal documents"""
    from transformers import (
        AutoModelForMaskedLM, DataCollatorForLanguageModeling,
        TrainingArguments, Trainer
    )
    
    # Convert model to MLM model if needed
    if not hasattr(model, 'cls'):
        model = AutoModelForMaskedLM.from_pretrained(model.name_or_path)
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_config['mlm_probability']
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=mlm_config['num_epochs'],
        per_device_train_batch_size=mlm_config['batch_size'],
        save_steps=mlm_config['save_steps'],
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=mlm_config['learning_rate'],
        warmup_steps=mlm_config['warmup_steps'],
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    print(f"🚀 Starting MLM pretraining...")
    print(f"  Epochs: {mlm_config['num_epochs']}")
    print(f"  Batch size: {mlm_config['batch_size']}")
    print(f"  Learning rate: {mlm_config['learning_rate']}")
    print(f"  Training samples: {len(train_dataset)}")
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ MLM pretraining completed!")
    print(f"💾 Model saved to: {output_dir}")
    
    return model, tokenizer


def validate_mlm_model(model_path: str, tokenizer_path: str, sample_texts: List[str]):
    """Validate MLM-adapted model by testing masked predictions"""
    from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
    
    # Load adapted model
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create fill-mask pipeline
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=5)
    
    print("🧪 Testing MLM-adapted model:")
    
    for i, text in enumerate(sample_texts[:3]):  # Test first 3 samples
        # Add mask to a legal term
        masked_text = text.replace("суд", tokenizer.mask_token, 1)
        
        print(f"\n📝 Sample {i+1}:")
        print(f"Original: {text[:100]}...")
        print(f"Masked: {masked_text[:100]}...")
        
        try:
            predictions = fill_mask(masked_text)
            print("🎯 Top predictions:")
            for pred in predictions:
                print(f"  {pred['token_str']}: {pred['score']:.4f}")
        except Exception as e:
            print(f"⚠️  Error in prediction: {e}")
    
    return model, tokenizer
