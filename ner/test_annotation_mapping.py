#!/usr/bin/env python3
"""
Test script to show the mapping between LabelStudio annotations and actual text parts
Updated to use actual LabelStudio files instead of local judgment files
"""

import json
import requests
from pathlib import Path

def download_labelstudio_file(file_path, token, cache_dir="labelstudio_cache"):
    """Download a file from LabelStudio storage with caching"""

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Extract filename for cache
    filename = file_path.split('/')[-1]
    cached_file = cache_path / filename

    # Check cache first
    if cached_file.exists():
        print(f"Using cached file: {cached_file}")
        with open(cached_file, 'r', encoding='utf-8') as f:
            return f.read()

    # Download from LabelStudio
    url = f"https://app.humansignal.com/storage-data/uploaded/?filepath={file_path}"
    headers = {"Authorization": f"Token {token}"}

    try:
        print(f"Downloading: {file_path}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        content = response.text

        # Cache the content
        with open(cached_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Downloaded and cached: {len(content)} characters")
        return content

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_path}: {e}")
        return None

def extract_judgment_filename(labelstudio_path):
    """Extract actual judgment filename from LabelStudio path"""
    actual_filename = labelstudio_path

    if 'upload/' in labelstudio_path and '-judgment_' in labelstudio_path:
        actual_filename = labelstudio_path.split('-')[-1]
        if not actual_filename.startswith('judgment_'):
            import re
            match = re.search(r'judgment_K_\d+_\d+\.txt', labelstudio_path)
            if match:
                actual_filename = match.group(0)

    return actual_filename

def test_annotation_mapping_with_labelstudio():
    """Test annotation mapping using actual LabelStudio files"""

    # Configuration
    TOKEN = "99cb57616d2c7b5b67da2d60d24dd5590605b89b"
    labelstudio_file = "export_186137_project-186137-at-2025-09-08-17-25-9356b6a3.json"

    if not Path(labelstudio_file).exists():
        print(f"Error: LabelStudio file not found: {labelstudio_file}")
        return

    # Load the data
    with open(labelstudio_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from LabelStudio export")

    # Find the specific item with file "5534cab7-judgment_K_959_2012.txt"
    target_file = "5534cab7-judgment_K_959_2012.txt"
    test_item = None

    for item in data:
        file_upload = item.get('file_upload', '')
        data_text = item.get('data', {}).get('text', '')

        if file_upload == target_file or target_file in data_text:
            test_item = item
            break

    if not test_item:
        print(f"Could not find item with file: {target_file}")
        print("Available files in first 5 items:")
        for i, item in enumerate(data[:5]):
            file_upload = item.get('file_upload', '')
            data_text = item.get('data', {}).get('text', '')
            print(f"  Item {i+1}: file_upload='{file_upload}', data.text='{data_text}'")
        return
    
    if not test_item:
        print("No annotated items found!")
        return
    
    # Get file path and load text
    file_path = test_item.get('data', {}).get('text', '')
    if not file_path:
        file_path = test_item.get('file_upload', '')

    print(f"Found target item with file_path: '{file_path}'")

    # Download the actual file from LabelStudio
    text_content = download_labelstudio_file(file_path, TOKEN)

    if not text_content:
        print(f"Error: Could not download file: {file_path}")
        return

    print(f"\nLabelStudio file loaded successfully")
    print(f"Length: {len(text_content)} characters")
    print(f"First 200 characters:")
    print(f"'{text_content[:200]}...'")

    # Get annotations
    annotations = test_item.get('annotations', [])
    if not annotations:
        print("No annotations found!")
        return
    
    annotation = annotations[0]
    results = annotation.get('result', [])
    
    print(f"\nFound {len(results)} annotation results:")
    print("=" * 80)
    
    # Process each annotation
    correct_count = 0
    total_count = 0

    for i, result in enumerate(results):
        if result.get('type') != 'labels':
            continue

        value = result.get('value', {})
        start = value.get('start')
        end = value.get('end')
        labels = value.get('labels', [])

        if start is None or end is None or not labels:
            continue

        # No offset needed - using actual LabelStudio files!
        # Skip if positions are invalid
        if start < 0 or end < 0 or start >= len(text_content) or end > len(text_content):
            continue

        total_count += 1

        # Extract the annotated text
        if start < len(text_content) and end <= len(text_content):
            annotated_text = text_content[start:end]
            label = labels[0]  # Take first label

            print(f"Annotation {i+1}:")
            print(f"  Label: {label}")
            print(f"  Position: {start}-{end}")
            print(f"  Text: '{annotated_text}'")
            print(f"  Length: {len(annotated_text)} characters")

            # Check if this looks correct (no partial words at start/end)
            is_correct = True
            if annotated_text and not annotated_text[0].isupper() and annotated_text[0].isalpha():
                is_correct = False
            if annotated_text.endswith(' ') or annotated_text.startswith(' '):
                is_correct = False

            if is_correct:
                correct_count += 1
                print(f"  Status: ✓ LOOKS CORRECT")
            else:
                print(f"  Status: ⚠ MIGHT BE INCORRECT")

            # Show context (50 chars before and after)
            context_start = max(0, start - 50)
            context_end = min(len(text_content), end + 50)
            context = text_content[context_start:context_end]

            # Mark the annotated part in context
            relative_start = start - context_start
            relative_end = end - context_start
            marked_context = (
                context[:relative_start] +
                ">>>" + context[relative_start:relative_end] + "<<<" +
                context[relative_end:]
            )

            print(f"  Context: '{marked_context}'")
        else:
            print(f"Annotation {i+1}: Position {start}-{end} out of range (text length: {len(text_content)})")

        print("-" * 60)
    
    # Show some statistics
    entity_counts = {}
    for result in results:
        if result.get('type') == 'labels':
            labels = result.get('value', {}).get('labels', [])
            for label in labels:
                entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print(f"\nEntity statistics for this file:")
    for entity, count in sorted(entity_counts.items()):
        print(f"  {entity}: {count}")

    print(f"\nSummary:")
    print(f"  Total annotations: {total_count}")
    print(f"  Correctly aligned: {correct_count}")
    if total_count > 0:
        print(f"  Accuracy: {correct_count/total_count*100:.1f}%")
    else:
        print("  No annotations to check")

    # Test known entities for perfect alignment
    print(f"\nTesting known entities for perfect alignment:")
    print("-" * 50)

    known_entities = {
        "COURT": "OSNOVNI SUD U NIKŠIĆU",
        "JUDGE": "Babović Dragan",
        "CASE_NUMBER": "959/12"
    }

    perfect_matches = 0
    total_known = 0

    for result in results:
        if result.get('type') != 'labels':
            continue

        value = result.get('value', {})
        start = value.get('start')
        end = value.get('end')
        labels = value.get('labels', [])

        if start is None or end is None or not labels:
            continue

        label = labels[0]

        if label in known_entities:
            total_known += 1
            expected_text = known_entities[label]

            if start < len(text_content) and end <= len(text_content):
                actual_text = text_content[start:end]

                print(f"{label}:")
                print(f"  Expected: '{expected_text}'")
                print(f"  Actual:   '{actual_text}'")
                print(f"  Position: {start}-{end}")

                if actual_text == expected_text:
                    print(f"  Status:   ✅ PERFECT MATCH!")
                    perfect_matches += 1
                elif expected_text in actual_text or actual_text in expected_text:
                    print(f"  Status:   ⚠️  Partial match")
                else:
                    print(f"  Status:   ❌ No match")
                print()

    print(f"Perfect matches: {perfect_matches}/{total_known}")

    if perfect_matches == total_known and total_known > 0:
        print("🎉 SUCCESS! All known entities match perfectly!")
        print("The LabelStudio files approach is working correctly.")
    elif perfect_matches > 0:
        print("✅ Partial success. Some entities match perfectly.")
    else:
        print("❌ No perfect matches. There may still be issues.")

def test_annotation_mapping():
    """Test annotation mapping using LabelStudio files"""
    test_annotation_mapping_with_labelstudio()

if __name__ == "__main__":
    test_annotation_mapping()
