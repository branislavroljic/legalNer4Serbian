#!/usr/bin/env python3
"""
Quick script to merge annotations.json and annotations_2.json
Simple version for immediate use.
"""

import json
import os
from datetime import datetime


def quick_merge():
    """Quick merge of annotation files with basic duplicate checking."""
    
    # File paths
    file1 = "ner/annotations.json"
    file2 = "ner/annotations_2.json"
    output = "ner/annotations_merged.json"
    backup = f"ner/annotations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print("🔄 Quick Annotation Merger")
    print("=" * 30)
    
    # Check if files exist
    if not os.path.exists(file1):
        print(f"❌ {file1} not found!")
        return
    if not os.path.exists(file2):
        print(f"❌ {file2} not found!")
        return
    
    try:
        # Load files
        print(f"📖 Loading {file1}...")
        with open(file1, 'r', encoding='utf-8') as f:
            annotations_1 = json.load(f)
        
        print(f"📖 Loading {file2}...")
        with open(file2, 'r', encoding='utf-8') as f:
            annotations_2 = json.load(f)
        
        print(f"✅ Loaded {len(annotations_1)} + {len(annotations_2)} annotations")
        
        # Create backup of original
        print(f"💾 Creating backup: {backup}")
        with open(backup, 'w', encoding='utf-8') as f:
            json.dump(annotations_1, f, ensure_ascii=False, indent=2)
        
        # Check for ID conflicts
        ids_1 = {item.get('id') for item in annotations_1 if 'id' in item}
        ids_2 = {item.get('id') for item in annotations_2 if 'id' in item}
        conflicts = ids_1.intersection(ids_2)
        
        if conflicts:
            print(f"⚠️  Found {len(conflicts)} ID conflicts - will skip duplicates")
            
            # Add only non-conflicting items
            added = 0
            for item in annotations_2:
                if item.get('id') not in ids_1:
                    annotations_1.append(item)
                    added += 1
            
            print(f"✅ Added {added} new annotations (skipped {len(annotations_2) - added} duplicates)")
        else:
            print("✅ No ID conflicts found - merging all")
            annotations_1.extend(annotations_2)
        
        # Save merged file
        print(f"💾 Saving merged file: {output}")
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(annotations_1, f, ensure_ascii=False, indent=2)
        
        print(f"🎉 Success! Merged file contains {len(annotations_1)} annotations")
        print(f"📁 Files created:")
        print(f"   - Merged: {output}")
        print(f"   - Backup: {backup}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    quick_merge()
