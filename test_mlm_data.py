#!/usr/bin/env python3
"""
Test script to verify MLM data loading for Domain-Adaptive Pretraining
"""

import os
from pathlib import Path

def test_mlm_data_loading():
    """Test if MLM documents can be loaded"""
    
    # Check MLM data directory
    mlm_dir = Path("mlm")
    
    if not mlm_dir.exists():
        print(f"❌ MLM directory not found: {mlm_dir}")
        print("Please ensure the 'mlm' folder exists in the current directory")
        return False
    
    # Count text files
    txt_files = list(mlm_dir.glob("*.txt"))
    print(f"📁 MLM directory: {mlm_dir.absolute()}")
    print(f"📄 Found {len(txt_files)} .txt files")
    
    if len(txt_files) == 0:
        print("❌ No .txt files found in MLM directory")
        return False
    
    # Test loading a few documents
    documents = []
    total_chars = 0
    
    for i, file_path in enumerate(txt_files[:5]):  # Test first 5 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
                    total_chars += len(content)
                    print(f"✅ Loaded {file_path.name}: {len(content)} chars")
                else:
                    print(f"⚠️  Empty file: {file_path.name}")
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
    
    if documents:
        print(f"\n📊 Sample Statistics:")
        print(f"   - Successfully loaded: {len(documents)} documents")
        print(f"   - Total characters: {total_chars:,}")
        print(f"   - Average length: {total_chars/len(documents):.0f} chars")
        
        # Show sample content
        print(f"\n📝 Sample content (first 200 chars):")
        print(f"   {documents[0][:200]}...")
        
        return True
    else:
        print("❌ No documents could be loaded")
        return False

if __name__ == "__main__":
    print("🧪 Testing MLM Data Loading for Domain-Adaptive Pretraining")
    print("=" * 60)
    
    success = test_mlm_data_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ MLM data loading test PASSED")
        print("💡 You can now run the notebook with Domain-Adaptive Pretraining")
    else:
        print("❌ MLM data loading test FAILED")
        print("💡 The notebook will skip DAPT and use the original model")
    
    print("\n📋 Next steps:")
    print("   1. Open the notebook: ner/serbian_legal_ner_pipeline_osnovni_editable.ipynb")
    print("   2. Run all cells to perform DAPT + NER training")
    print("   3. The pipeline will automatically detect and use MLM data if available")
