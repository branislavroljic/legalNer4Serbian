#!/usr/bin/env python3
"""
Pure Playwright scraper for Montenegro court decisions
Handles SPA properly by staying in the same browser context
"""

import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright


class MontenegroCourtscraper:
    def __init__(self, output_dir="../judgments"):
        self.base_url = "https://sudovi.me/sdvi/odluke"
        self.search_term = "kriv je"
        self.max_results = 100
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
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
    
    async def perform_search(self, page):
        """Perform the search for 'kriv je'"""
        print("Performing search...")
        
        # Click "Napredna pretraga" tab
        print("Clicking 'Napredna pretraga' tab...")
        await page.click('div.q-tab:has-text("Napredna pretraga")')
        await page.wait_for_timeout(1000)
        
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
        results_to_process = result_links[:self.max_results]
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
                # Save to file
                filename = await self.save_judgment(
                    court_name, case_number, date, judgment_text, index
                )
                print(f"Saved judgment to: {filename}")
            else:
                print("No judgment text found!")

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
        """Save judgment to a text file"""
        # Create safe filename
        if case_number and case_number != 'unknown':
            safe_case_number = case_number.replace('/', '_').replace(' ', '_')
            filename = f"judgment_{safe_case_number}.txt"
        else:
            filename = f"judgment_result_{index}.txt"

        # Full path to output file
        filepath = self.output_dir / filename

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Court: {court_name}\n")
            f.write(f"Case Number: {case_number}\n")
            f.write(f"Date: {date}\n")
            f.write("=" * 50 + "\n\n")
            f.write(judgment_text)

        return filepath


async def main():
    """Main function to run the scraper"""
    scraper = MontenegroCourtscraper()
    await scraper.run()


if __name__ == "__main__":
    print("Starting Montenegro Court Scraper")
    print("=" * 50)
    asyncio.run(main())
    print("\nScraping completed!")
