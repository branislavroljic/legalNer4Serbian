#!/usr/bin/env python3
"""
Download the actual files that LabelStudio used for annotations
"""

import json
import requests
from pathlib import Path
import os

def download_labelstudio_file(file_path, token, output_dir="labelstudio_files_2"):
    """Download a file from LabelStudio storage"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the URL
    base_url = "https://app.humansignal.com/storage-data/uploaded/"
    url = f"{base_url}?filepath={file_path}"
    
    # Set headers
    headers = {
        "Authorization": f"Token {token}"
    }
    
    try:
        print(f"Downloading: {file_path}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Extract filename from file_path
        filename = file_path.split('/')[-1]
        output_path = Path(output_dir) / filename
        
        # Save the content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"  Saved to: {output_path}")
        print(f"  Length: {len(response.text)} characters")
        
        return output_path, response.text
        
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {file_path}: {e}")
        return None, None

def download_all_labelstudio_files():
    """Download all files referenced in the LabelStudio export"""
    
    # Configuration
    TOKEN = "d80066b95a52516067abd4f045ebb192049b0f8b"
    labelstudio_file = "annotations_2.json"
    
    if not Path(labelstudio_file).exists():
        print(f"Error: LabelStudio file not found: {labelstudio_file}")
        return
    
    # Load the data
    with open(labelstudio_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from LabelStudio export")
    
    # Extract unique file paths
    file_paths = set()
    for item in data:
        # Check both possible locations for file path
        file_path = item.get('data', {}).get('text', '')
        if not file_path:
            file_path = item.get('file_upload', '')
        
        if file_path:
            file_paths.add(file_path)
    
    print(f"Found {len(file_paths)} unique files to download")
    
    # Download each file
    downloaded_files = {}
    for file_path in sorted(file_paths):
        output_path, content = download_labelstudio_file(file_path, TOKEN)
        if output_path and content:
            downloaded_files[file_path] = {
                'local_path': output_path,
                'content': content
            }
    
    print(f"\nSuccessfully downloaded {len(downloaded_files)} files")
    return downloaded_files

def compare_with_local_files(downloaded_files):
    """Compare downloaded files with local judgment files"""
    
    judgments_dir = Path("../judgments")
    
    if not judgments_dir.exists():
        print(f"Local judgments directory not found: {judgments_dir}")
        return
    
    print(f"\nComparing with local files in {judgments_dir}")
    print("=" * 60)
    
    for file_path, file_info in downloaded_files.items():
        # Extract actual filename
        if '-judgment_' in file_path:
            actual_filename = file_path.split('-')[-1]
        else:
            actual_filename = file_path.split('/')[-1]
        
        local_file_path = judgments_dir / actual_filename
        labelstudio_content = file_info['content']
        
        print(f"\nFile: {actual_filename}")
        print(f"  LabelStudio path: {file_path}")
        print(f"  Local path: {local_file_path}")
        print(f"  LabelStudio length: {len(labelstudio_content)}")
        
        if local_file_path.exists():
            # Load local file
            with open(local_file_path, 'r', encoding='utf-8-sig') as f:
                local_content = f.read()
            
            print(f"  Local length: {len(local_content)}")
            
            # Compare
            if labelstudio_content == local_content:
                print(f"  Status: ✓ IDENTICAL")
            else:
                print(f"  Status: ✗ DIFFERENT")
                
                # Show differences
                ls_lines = labelstudio_content.split('\n')
                local_lines = local_content.split('\n')
                
                print(f"  LabelStudio lines: {len(ls_lines)}")
                print(f"  Local lines: {len(local_lines)}")
                
                # Show first few lines
                print(f"  LabelStudio first line: {repr(ls_lines[0] if ls_lines else '')}")
                print(f"  Local first line: {repr(local_lines[0] if local_lines else '')}")
                
                # Check if one is a subset of the other
                if labelstudio_content in local_content:
                    print(f"  LabelStudio content is subset of local")
                elif local_content in labelstudio_content:
                    print(f"  Local content is subset of LabelStudio")
        else:
            print(f"  Status: ✗ LOCAL FILE NOT FOUND")

def test_annotation_with_labelstudio_file():
    """Test annotation alignment using the actual LabelStudio file"""
    
    # Download the specific test file
    TOKEN = "d80066b95a52516067abd4f045ebb192049b0f8b"
    test_file_path = "upload/189142/8296792b-judgment_K_564_2022.txt"
    
    output_path, content = download_labelstudio_file(test_file_path, TOKEN)
    
    if not content:
        print("Failed to download test file")
        return
    
    # Load LabelStudio annotations
    labelstudio_file = "annotations_2.json"
    with open(labelstudio_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find the test item
    target_file = "8296792b-judgment_K_564_2022.txt"
    test_item = None
    
    for item in data:
        file_upload = item.get('file_upload', '')
        data_text = item.get('data', {}).get('text', '')
        
        if file_upload == target_file or target_file in data_text:
            test_item = item
            break
    
    if not test_item:
        print(f"Could not find annotations for {target_file}")
        return
    
    # Test annotations
    annotations = test_item.get('annotations', [])
    if not annotations:
        print("No annotations found!")
        return
    
    annotation = annotations[0]
    results = annotation.get('result', [])
    
    print(f"\nTesting annotations with LabelStudio file:")
    print(f"File length: {len(content)} characters")
    print("=" * 60)
    
    correct_count = 0
    total_count = 0
    
    for i, result in enumerate(results[:10]):  # Test first 10
        if result.get('type') != 'labels':
            continue
        
        value = result.get('value', {})
        start = value.get('start')
        end = value.get('end')
        labels = value.get('labels', [])
        
        if start is None or end is None or not labels:
            continue
        
        total_count += 1
        label = labels[0]
        
        if start < len(content) and end <= len(content):
            annotated_text = content[start:end]
            
            print(f"Annotation {i+1}:")
            print(f"  Label: {label}")
            print(f"  Position: {start}-{end}")
            print(f"  Text: '{annotated_text}'")
            
            # Check if it looks correct
            if (annotated_text and 
                not annotated_text.startswith(' ') and 
                not annotated_text.endswith(' ') and
                (annotated_text[0].isupper() or not annotated_text[0].isalpha())):
                correct_count += 1
                print(f"  Status: ✓ LOOKS CORRECT")
            else:
                print(f"  Status: ⚠ MIGHT BE INCORRECT")
            
            print()
    
    print(f"Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    # print("Downloading LabelStudio files...")
    # downloaded_files = download_all_labelstudio_files()
    
    # if downloaded_files:
    #     compare_with_local_files(downloaded_files)
    #     print("\n" + "="*60)
    test_annotation_with_labelstudio_file()
    # else:
    #     print("No files were downloaded successfully")
