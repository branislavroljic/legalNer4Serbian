#!/usr/bin/env python3
"""
Script to merge LabelStudio annotation files.
Merges annotations_2.json into annotations.json with duplicate detection.
"""

import json
import os
import argparse
from typing import List, Dict, Set, Any
import hashlib


def calculate_text_hash(text: str) -> str:
    """Calculate a hash for text content to detect duplicates."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_annotations(file_path: str) -> List[Dict[str, Any]]:
    """Load annotations from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"✓ Loaded {len(annotations)} annotations from {file_path}")
        return annotations
    except FileNotFoundError:
        print(f"✗ Error: File {file_path} not found!")
        return []
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in {file_path}: {e}")
        return []


def analyze_annotations(annotations: List[Dict[str, Any]], file_name: str) -> Dict[str, Any]:
    """Analyze annotation structure and content."""
    if not annotations:
        return {}
    
    analysis = {
        'total_count': len(annotations),
        'unique_ids': set(),
        'text_hashes': set(),
        'file_uploads': set(),
        'projects': set(),
        'sample_annotation': annotations[0] if annotations else None
    }
    
    for annotation in annotations:
        # Collect unique identifiers
        if 'id' in annotation:
            analysis['unique_ids'].add(annotation['id'])
        
        # Collect project IDs
        if 'project' in annotation:
            analysis['projects'].add(annotation['project'])
        
        # Collect file uploads
        if 'file_upload' in annotation:
            analysis['file_uploads'].add(annotation['file_upload'])
        
        # Calculate text hash for duplicate detection
        if 'data' in annotation and 'text' in annotation['data']:
            text_hash = calculate_text_hash(annotation['data']['text'])
            analysis['text_hashes'].add(text_hash)
    
    print(f"\n📊 Analysis of {file_name}:")
    print(f"   Total annotations: {analysis['total_count']}")
    print(f"   Unique IDs: {len(analysis['unique_ids'])}")
    print(f"   Unique texts: {len(analysis['text_hashes'])}")
    print(f"   Projects: {analysis['projects']}")
    print(f"   File uploads: {len(analysis['file_uploads'])}")
    
    return analysis


def merge_annotations_simple(annotations_1: List[Dict], annotations_2: List[Dict]) -> List[Dict]:
    """Simple concatenation merge."""
    return annotations_1 + annotations_2


def merge_annotations_safe(annotations_1: List[Dict], annotations_2: List[Dict]) -> tuple[List[Dict], Dict]:
    """Merge with duplicate detection."""
    # Track existing items
    existing_ids = {item['id'] for item in annotations_1 if 'id' in item}
    existing_text_hashes = set()
    existing_file_uploads = set()
    
    # Build existing content tracking
    for item in annotations_1:
        if 'file_upload' in item:
            existing_file_uploads.add(item['file_upload'])
        if 'data' in item and 'text' in item['data']:
            text_hash = calculate_text_hash(item['data']['text'])
            existing_text_hashes.add(text_hash)
    
    # Merge process
    merged = annotations_1.copy()
    stats = {
        'added': 0,
        'duplicate_ids': 0,
        'duplicate_texts': 0,
        'duplicate_files': 0
    }
    
    for item in annotations_2:
        skip_reasons = []
        
        # Check for ID duplicates
        if 'id' in item and item['id'] in existing_ids:
            skip_reasons.append('duplicate_id')
            stats['duplicate_ids'] += 1
        
        # Check for text content duplicates
        if 'data' in item and 'text' in item['data']:
            text_hash = calculate_text_hash(item['data']['text'])
            if text_hash in existing_text_hashes:
                skip_reasons.append('duplicate_text')
                stats['duplicate_texts'] += 1
        
        # Check for file upload duplicates
        if 'file_upload' in item and item['file_upload'] in existing_file_uploads:
            skip_reasons.append('duplicate_file')
            stats['duplicate_files'] += 1
        
        # Add if no duplicates found
        if not skip_reasons:
            merged.append(item)
            stats['added'] += 1
            
            # Update tracking sets
            if 'id' in item:
                existing_ids.add(item['id'])
            if 'file_upload' in item:
                existing_file_uploads.add(item['file_upload'])
            if 'data' in item and 'text' in item['data']:
                text_hash = calculate_text_hash(item['data']['text'])
                existing_text_hashes.add(text_hash)
    
    return merged, stats


def save_annotations(annotations: List[Dict], output_path: str) -> bool:
    """Save annotations to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(annotations)} annotations to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving to {output_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Merge LabelStudio annotation files')
    parser.add_argument('--input1', default='ner/annotations.json', 
                       help='First annotation file (default: ner/annotations.json)')
    parser.add_argument('--input2', default='ner/annotations_2.json',
                       help='Second annotation file (default: ner/annotations_2.json)')
    parser.add_argument('--output', default='ner/annotations_merged.json',
                       help='Output file (default: ner/annotations_merged.json)')
    parser.add_argument('--mode', choices=['simple', 'safe'], default='safe',
                       help='Merge mode: simple (concatenate) or safe (check duplicates)')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of original files')
    
    args = parser.parse_args()
    
    print("🔄 LabelStudio Annotation Merger")
    print("=" * 40)
    
    # Load annotation files
    annotations_1 = load_annotations(args.input1)
    annotations_2 = load_annotations(args.input2)
    
    if not annotations_1 or not annotations_2:
        print("✗ Cannot proceed without both annotation files.")
        return
    
    # Analyze files
    analysis_1 = analyze_annotations(annotations_1, args.input1)
    analysis_2 = analyze_annotations(annotations_2, args.input2)
    
    # Create backups if requested
    if args.backup:
        backup_1 = f"{args.input1}.backup"
        backup_2 = f"{args.input2}.backup"
        save_annotations(annotations_1, backup_1)
        save_annotations(annotations_2, backup_2)
        print(f"✓ Created backups: {backup_1}, {backup_2}")
    
    # Merge annotations
    print(f"\n🔀 Merging using '{args.mode}' mode...")
    
    if args.mode == 'simple':
        merged = merge_annotations_simple(annotations_1, annotations_2)
        print(f"✓ Simple merge completed: {len(annotations_1)} + {len(annotations_2)} = {len(merged)}")
    else:
        merged, stats = merge_annotations_safe(annotations_1, annotations_2)
        print(f"✓ Safe merge completed:")
        print(f"   Added: {stats['added']} new annotations")
        print(f"   Skipped duplicates: ID={stats['duplicate_ids']}, Text={stats['duplicate_texts']}, File={stats['duplicate_files']}")
        print(f"   Total result: {len(merged)} annotations")
    
    # Save merged file
    if save_annotations(merged, args.output):
        print(f"\n🎉 Merge successful! Output saved to: {args.output}")
        
        # Final verification
        verification = load_annotations(args.output)
        if len(verification) == len(merged):
            print("✓ File verification passed")
        else:
            print("⚠ Warning: File verification failed")
    else:
        print("✗ Merge failed during save operation")


if __name__ == "__main__":
    main()
