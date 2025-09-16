#!/usr/bin/env python3
"""
CSS Cleaner for Montenegro Court Judgment Files
Removes CSS styling and HTML artifacts from scraped judgment texts
"""

import re
import os
from pathlib import Path


class CSSCleaner:
    def __init__(self):
        """Initialize CSS cleaner with common patterns"""
        
        # CSS class definitions pattern
        self.css_class_pattern = re.compile(
            r'\.[\w\d]+\{[^}]*\}',
            re.MULTILINE | re.DOTALL
        )
        
        # HTML/CSS artifacts patterns
        self.html_patterns = [
            # CSS class definitions like .b1{white-space-collapsing:preserve;}
            re.compile(r'\.[\w\d]+\{[^}]*\}', re.MULTILINE | re.DOTALL),
            
            # HTML tags
            re.compile(r'<[^>]+>', re.MULTILINE),
            
            # CSS properties in text
            re.compile(r'(white-space-collapsing|background|margin|width|padding-top|color|font-family|font-weight|font-size):[^;]*;?', re.MULTILINE),
            
            # Standalone CSS values
            re.compile(r'(#fff|#[0-9A-Fa-f]{6}|#[0-9A-Fa-f]{3})', re.MULTILINE),
            
            # CSS units
            re.compile(r'\d+(\.\d+)?(mm|pt|px|em|rem|%|in)', re.MULTILINE),
            
            # CSS keywords
            re.compile(r'\b(preserve|auto|Arial)\b', re.MULTILINE),
            
            # Multiple spaces and tabs
            re.compile(r'[ \t]+', re.MULTILINE),
            
            # Multiple newlines (more than 2)
            re.compile(r'\n{3,}', re.MULTILINE),
        ]
    
    def clean_text(self, text):
        """Clean CSS and HTML artifacts from text"""
        if not text:
            return text
        
        cleaned_text = text
        
        # Apply all cleaning patterns
        for pattern in self.html_patterns:
            cleaned_text = pattern.sub(' ', cleaned_text)
        
        # Clean up whitespace
        lines = cleaned_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if line and not self._is_css_line(line):
                cleaned_lines.append(line)
        
        # Join lines and clean up spacing
        result = '\n'.join(cleaned_lines)
        
        # Final cleanup
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
        result = re.sub(r'[ \t]+', ' ', result)     # Single spaces only
        
        return result.strip()
    
    def _is_css_line(self, line):
        """Check if a line contains only CSS artifacts"""
        css_indicators = [
            'white-space-collapsing',
            'background:#fff',
            'margin:0',
            'font-family:',
            'font-weight:',
            'font-size:',
            'color:#',
            '.span',
            '.window',
            '.page',
            '.b1{',
        ]
        
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in css_indicators)
    
    def clean_file(self, file_path, remove_header=True):
        """Clean a single judgment file"""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if remove_header:
                # Remove header completely - find content after separator
                parts = content.split('=' * 20)
                if len(parts) > 1:
                    # Take everything after the separator
                    body_text = parts[1]
                else:
                    # No separator found, clean entire content
                    body_text = content

                # Clean the body text
                cleaned_content = self.clean_text(body_text)
            else:
                # Keep header, clean only body
                lines = content.split('\n')
                header_lines = []
                body_lines = []
                separator_found = False

                for line in lines:
                    if '=' * 20 in line:  # Find the separator
                        header_lines.append(line)
                        separator_found = True
                    elif not separator_found:
                        header_lines.append(line)
                    else:
                        body_lines.append(line)

                # Clean only the body (judgment text)
                body_text = '\n'.join(body_lines)
                cleaned_body = self.clean_text(body_text)

                # Reconstruct the file
                header_text = '\n'.join(header_lines)
                cleaned_content = header_text + '\n\n' + cleaned_body

            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            print(f"✅ Cleaned: {file_path}")
            return True

        except Exception as e:
            print(f"❌ Error cleaning {file_path}: {e}")
            return False
    
    def clean_directory(self, directory_path, remove_header=True):
        """Clean all judgment files in a directory"""
        directory = Path(directory_path)

        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return

        # Find all judgment files
        judgment_files = list(directory.glob("judgment_*.txt"))

        if not judgment_files:
            print(f"❌ No judgment files found in: {directory}")
            return

        header_msg = "with header removal" if remove_header else "keeping headers"
        print(f"Found {len(judgment_files)} judgment files to clean ({header_msg})...")

        cleaned_count = 0
        for file_path in judgment_files:
            if self.clean_file(file_path, remove_header=remove_header):
                cleaned_count += 1

        print(f"\n✅ Successfully cleaned {cleaned_count}/{len(judgment_files)} files")
    
    def preview_cleaning(self, file_path, lines=10):
        """Preview what cleaning would do to a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the judgment text (after separator)
            parts = content.split('=' * 50)
            if len(parts) > 1:
                original_text = parts[1][:1000]  # First 1000 chars
                cleaned_text = self.clean_text(original_text)
                
                print("ORIGINAL TEXT (first 1000 chars):")
                print("-" * 40)
                print(original_text)
                print("\nCLEANED TEXT:")
                print("-" * 40)
                print(cleaned_text)
            else:
                print("❌ Could not find judgment text in file")
                
        except Exception as e:
            print(f"❌ Error previewing {file_path}: {e}")


def main():
    """Main function with command line interface"""
    import sys

    cleaner = CSSCleaner()

    if len(sys.argv) < 2:
        print("CSS Cleaner for Montenegro Court Judgments")
        print("=" * 45)
        print("Usage:")
        print("  python css_cleaner.py <directory>              # Clean all files (remove headers)")
        print("  python css_cleaner.py <file>                   # Clean single file (remove header)")
        print("  python css_cleaner.py <directory> --keep-header # Clean all files (keep headers)")
        print("  python css_cleaner.py <file> --keep-header     # Clean single file (keep header)")
        print("  python css_cleaner.py preview <file>           # Preview cleaning")
        print("\nExamples:")
        print("  python css_cleaner.py ../judgments")
        print("  python css_cleaner.py ../judgments --keep-header")
        print("  python css_cleaner.py ../judgments/judgment_K_224_2011.txt")
        print("  python css_cleaner.py preview ../judgments/judgment_K_224_2011.txt")
        return

    # Check for options
    remove_header = True
    if "--keep-header" in sys.argv:
        remove_header = False
        sys.argv.remove("--keep-header")

    if sys.argv[1] == "preview" and len(sys.argv) > 2:
        # Preview mode
        file_path = Path(sys.argv[2])
        if file_path.exists():
            cleaner.preview_cleaning(file_path)
        else:
            print(f"❌ File not found: {file_path}")
    else:
        # Clean mode
        target_path = Path(sys.argv[1])

        if target_path.is_file():
            # Clean single file
            cleaner.clean_file(target_path, remove_header=remove_header)
        elif target_path.is_dir():
            # Clean directory
            cleaner.clean_directory(target_path, remove_header=remove_header)
        else:
            print(f"❌ Path not found: {target_path}")


if __name__ == "__main__":
    main()
