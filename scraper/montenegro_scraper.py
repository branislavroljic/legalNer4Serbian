#!/usr/bin/env python3
"""
Pure Playwright scraper for Montenegro court decisions
Handles SPA properly by staying in the same browser context
"""

import asyncio
import os
import re
from pathlib import Path
from playwright.async_api import async_playwright


class MontenegroCourtscraper:
    def __init__(self, output_dir="../judgments"):
        self.base_url = "https://sudovi.me/sdvi/odluke"
        self.search_term = "kriv je"
        self.max_results = 30
        self.output_dir = Path(output_dir).resolve()  # Get absolute path

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def clean_judgment_text(self, text):
        """Clean up judgment text by removing CSS, HTML and unnecessary formatting"""
        if not text:
            return text

        # Remove CSS class definitions
        text = re.sub(r'\.[\w\d]+\{[^}]*\}', '', text, flags=re.MULTILINE | re.DOTALL)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove CSS properties
        css_properties = [
            r'white-space-collapsing:[^;]*;?',
            r'background:[^;]*;?',
            r'margin:[^;]*;?',
            r'width:[^;]*;?',
            r'padding-top:[^;]*;?',
            r'color:[^;]*;?',
            r'font-family:[^;]*;?',
            r'font-weight:[^;]*;?',
            r'font-size:[^;]*;?'
        ]

        for prop_pattern in css_properties:
            text = re.sub(prop_pattern, '', text, flags=re.MULTILINE)

        # Remove CSS values and units
        text = re.sub(r'#[0-9A-Fa-f]{3,6}', '', text)  # Color codes
        text = re.sub(r'\d+(\.\d+)?(mm|pt|px|em|rem|%|in)', '', text)  # CSS units

        # Split into lines and process
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Strip whitespace from each line
            line = line.strip()

            # Skip empty lines and CSS-only lines
            if not line or self._is_css_line(line):
                continue

            # Add the cleaned line
            cleaned_lines.append(line)

        # Join lines with single newlines, but preserve paragraph breaks
        result = []
        for line in cleaned_lines:
            result.append(line)

            # Add extra newline after certain patterns (paragraph breaks)
            if (line.endswith('.') or line.endswith(':') or
                line.startswith('OSNOVNI SUD') or line.startswith('PRESUDU') or
                line.startswith('OKRIVLJENI') or line.startswith('Kriv je') or
                line.startswith('USLOVNU OSUDU')):
                result.append('')  # Add empty line for paragraph break

        # Join and clean up multiple consecutive newlines
        text = '\n'.join(result)

        # Remove more than 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Clean up extra spaces
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

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
            'preserve',
            'Arial'
        ]

        line_lower = line.lower()
        return any(indicator in line_lower for indicator in css_indicators)
    async def run(self):
        """Main scraper method"""
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=False,  # Set to True for headless mode
            )
            
            # Create new page
            page = await browser.new_page()
            
            try:
                # Navigate to the site
                print(f"Navigating to {self.base_url}")
                await page.goto(self.base_url)
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)
                
                # Handle cookie consent
                await self.handle_cookies(page)
                
                # Perform search
                await self.perform_search(page)
                
                # Extract and process results
                await self.process_results(page)
                
            except Exception as e:
                print(f"Error during scraping: {e}")
            finally:
                # Close browser
                await browser.close()
    
    async def handle_cookies(self, page):
        """Handle cookie consent dialog if present"""
        print("Checking for cookie dialog...")
        
        try:
            # Wait a bit for dialog to appear
            await page.wait_for_timeout(1000)
            
            # Check if cookie dialog exists
            cookie_dialog = await page.query_selector('.gdpr-dialog')
            if cookie_dialog:
                print("Cookie dialog found, clicking accept...")
                accept_button = await page.query_selector('button.gdpr-button-accept')
                if accept_button:
                    await accept_button.click()
                    await page.wait_for_timeout(1000)
                    print("Cookie dialog accepted")
                else:
                    print("Accept button not found")
            else:
                print("No cookie dialog found")
        except Exception as e:
            print(f"Error handling cookies: {e}")

    async def select_presuda_option(self, page):
        """Select 'Presuda' option from the decision type dropdown"""
        try:
            # Look for the combobox with "Presuda" text
            combobox_selector = 'div[role="combobox"][aria-label="Vrsta odluke"]'

            # Wait for the combobox to be available
            await page.wait_for_selector(combobox_selector, timeout=5000)

            # Click on the combobox to open dropdown
            await page.click(combobox_selector)
            await page.wait_for_timeout(500)

            # Wait for dropdown options to appear and click "Presuda"
            # Try different possible selectors for the Presuda option
            presuda_selectors = [
                'div.q-item:has-text("Presuda")',
                'div[role="option"]:has-text("Presuda")',
                '.q-menu div:has-text("Presuda")',
                'div.q-item-label:has-text("Presuda")'
            ]

            option_clicked = False
            for selector in presuda_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=2000)
                    await page.click(selector)
                    option_clicked = True
                    print("Successfully selected 'Presuda' option")
                    break
                except:
                    continue

            if not option_clicked:
                print("Warning: Could not find 'Presuda' option in dropdown")

            await page.wait_for_timeout(500)

        except Exception as e:
            print(f"Error selecting Presuda option: {e}")

    async def perform_search(self, page):
        """Perform the search for 'kriv je'"""
        print("Performing search...")
        
        # Click "Napredna pretraga" tab
        print("Clicking 'Napredna pretraga' tab...")
        await page.click('div.q-tab:has-text("Napredna pretraga")')
        await page.wait_for_timeout(1000)

        # Select "Presuda" from the decision type dropdown
        print("Selecting 'Presuda' from decision type dropdown...")
        await self.select_presuda_option(page)

        # Fill search field
        print(f"Filling search field with '{self.search_term}'...")
        await page.fill('input[aria-label="Pretraga odluke"]', self.search_term)
        await page.wait_for_timeout(500)
        
        # Click search button
        print("Clicking search button...")
        await page.click('span.q-btn__content:has-text("Pretraga")')
        
        # Wait for results to load
        print("Waiting for search results...")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(3000)
        
        # Wait for result links to appear
        await page.wait_for_selector('a[data-v-27f7abb3]', timeout=10000)
        print("Search results loaded successfully")
    
    async def process_results(self, page):
        """Extract and process the first N results"""
        print(f"Processing first {self.max_results} results...")
        
        # Get all result links
        result_links = await page.query_selector_all('a[data-v-27f7abb3]')
        
        if not result_links:
            print("No search results found!")
            return
        
        # Process only the first N results
        results_to_process = result_links[400:400+self.max_results]
        print(f"Found {len(result_links)} total results, processing first {len(results_to_process)}")
        
        for i, result_link in enumerate(results_to_process):
            await self.process_single_result(page, result_link, i + 1)
    
    async def process_single_result(self, page, result_link, index):
        """Process a single search result"""
        print(f"\n--- Processing Result {index} ---")
        
        try:
            # Extract metadata from the result link
            court_name = await self.extract_text(result_link, 'label')
            case_number = await self.extract_text(result_link, 'span.center span')
            date = await self.extract_text(result_link, 'label.lower span')
            
            print(f"Court: {court_name}")
            print(f"Case: {case_number}")
            print(f"Date: {date}")
            
            # Click on the result to open judgment (opens new window)
            print("Clicking on result...")
            
            # Listen for new page/window
            async with page.context.expect_page() as new_page_info:
                await result_link.click()
            
            # Get the new page that opened
            new_page = await new_page_info.value
            
            # Wait for the new page to load
            print("Waiting for judgment page to load in new window...")
            await new_page.wait_for_load_state("networkidle")
            await new_page.wait_for_timeout(2000)
            
            # Try to find the judgment content in the new page
            judgment_text = await self.extract_judgment_text(new_page)
            
            # Close the new page after extracting content
            await new_page.close()

            if judgment_text:
                print(f"Extracted {len(judgment_text)} characters of judgment text")
                # Save to file
                filename = await self.save_judgment(
                    court_name, case_number, date, judgment_text, index
                )
                if filename:
                    print(f"✅ Saved judgment to: {filename}")
                else:
                    print("❌ Failed to save judgment")
            else:
                print("❌ No judgment text found!")

            # Go back to search results (if needed)
            # For SPA, we might need to navigate back or the results might still be visible
            await page.wait_for_timeout(1000)

        except Exception as e:
            print(f"Error processing result {index}: {e}")

    async def extract_text(self, element, selector):
        """Extract text from an element using a selector"""
        try:
            sub_element = await element.query_selector(selector)
            if sub_element:
                text = await sub_element.text_content()
                return text.strip() if text else 'unknown'
            return 'unknown'
        except:
            return 'unknown'

    async def extract_judgment_text(self, page):
        """Extract judgment text from the current page"""
        try:
            # Wait for judgment content to load
            await page.wait_for_selector('article[data-v-7a7c2765]', timeout=5000)

            # Extract all text from the article
            article = await page.query_selector('article[data-v-7a7c2765]')
            if article:
                judgment_text = await article.text_content()
                return judgment_text.strip() if judgment_text else None

            # Fallback: try alternative selector
            article = await page.query_selector('article')
            if article:
                judgment_text = await article.text_content()
                return judgment_text.strip() if judgment_text else None

            return None

        except Exception as e:
            print(f"Error extracting judgment text: {e}")
            return None

    async def save_judgment(self, court_name, case_number, date, judgment_text, index):
        """Save judgment to a text file with plain text only (no headers)"""
        try:
            # Clean the judgment text
            cleaned_text = self.clean_judgment_text(judgment_text)

            # Create safe filename
            if case_number and case_number != 'unknown':
                safe_case_number = case_number.replace('/', '_').replace(' ', '_')
                filename = f"judgment_{safe_case_number}.txt"
            else:
                filename = f"judgment_result_{index}.txt"

            # Full path to output file
            filepath = self.output_dir / filename

            print(f"Saving to: {filepath}")

            # Write to file - only the cleaned judgment text, no headers
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            # Verify file was created
            if filepath.exists():
                file_size = filepath.stat().st_size
                print(f"File saved successfully: {filepath} ({file_size} bytes)")
            else:
                print(f"ERROR: File was not created: {filepath}")

            return filepath

        except Exception as e:
            print(f"ERROR saving judgment {index}: {e}")
            return None


async def main():
    """Main function to run the scraper"""
    scraper = MontenegroCourtscraper()
    await scraper.run()


if __name__ == "__main__":
    print("Starting Montenegro Court Scraper")
    print("=" * 50)
    asyncio.run(main())
    print("\nScraping completed!")
