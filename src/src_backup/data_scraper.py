import os
import requests
import json
import chardet
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import logging
import argparse
import random
import yaml

class GutenbergScraper:
    def __init__(self, output_dir="data", cache_dir="cache"):
        self.base_url = "https://www.gutenberg.org"
        self.output_dir = os.path.join(output_dir, 'gutenberg')
        self.cache_dir = os.path.join(cache_dir, 'gutenberg')
        self.session = requests.Session()
        
        # Setup directories
        for dir in [self.output_dir, self.cache_dir]:
            os.makedirs(dir, exist_ok=True)
            os.makedirs(os.path.join(dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(dir, 'test'), exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        self.logger = logging.getLogger('GutenbergScraper')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('scraping.log')
        
        format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(format)
        f_handler.setFormatter(format)
        
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def get_bookshelf_books(self, bookshelf_id):
        """Get list of books from a bookshelf."""
        cache_file = os.path.join(self.cache_dir, f"bookshelf_{bookshelf_id}_books.json")
        
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached book list for bookshelf: {bookshelf_id}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        books = []
        page_index = 0
        consecutive_failures = 0
        max_failures = 5
        
        while consecutive_failures < max_failures:
            url = f"{self.base_url}/ebooks/bookshelf/{bookshelf_id}"
            if page_index > 0:
                url += f"?start_index={page_index * 25 + 1}"
            
            self.logger.info(f"Fetching page {page_index + 1} from {url}")
            response = self.session.get(url)
            
            if response.status_code != 200:
                consecutive_failures += 1
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            book_elements = soup.select('li.booklink')
            
            if not book_elements:
                break
                
            for book in book_elements:
                title_elem = book.select_one('span.title')
                author_elem = book.select_one('span.subtitle')
                link_elem = book.select_one('a[href*="/ebooks/"]')
                
                if title_elem and link_elem:
                    book_id = link_elem['href'].split('/')[-1]
                    books.append({
                        'id': book_id,
                        'title': title_elem.text.strip(),
                        'author': author_elem.text.strip() if author_elem else "Unknown",
                        'url': f"{self.base_url}/ebooks/{book_id}"
                    })
            
            page_index += 1
            time.sleep(1)
        
        # Cache results
        with open(cache_file, 'w') as f:
            json.dump(books, f, indent=2)
        
        self.logger.info(f"Found {len(books)} books in bookshelf: {bookshelf_id}")
        return books

    def download_book(self, book_id):
        """Download a book by its ID."""
        cache_file = os.path.join(self.cache_dir, f"book_{book_id}.txt")
        
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached book: {book_id}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        txt_url = f"{self.base_url}/cache/epub/{book_id}/pg{book_id}.txt"
        response = self.session.get(txt_url)
        
        if response.status_code != 200:
            self.logger.warning(f"Failed to download book {book_id}, trying alternate URL")
            txt_url = f"{self.base_url}/files/{book_id}/pg{book_id}.txt"
            response = self.session.get(txt_url)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to download book {book_id}")
                return None
        
        encoding = chardet.detect(response.content)['encoding']
        text = response.content.decode(encoding or 'utf-8', errors='ignore')
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return text

    def process_books(self, bookshelf_id, test_size=2):
        """Process all books from a bookshelf and split into train/test."""
        books = self.get_bookshelf_books(bookshelf_id)
        random.shuffle(books)  # Randomize before splitting
        
        # Split into train and test
        test_books = books[:test_size]
        train_books = books[test_size:]
        
        # Process test books
        test_texts = []
        successful_test_books = []
        test_positions = []
        current_position = 0
        
        # Save files
        train_file = os.path.join(self.output_dir, 'train', f"bookshelf_{bookshelf_id}_train.jsonl")
        test_file = os.path.join(self.output_dir, 'test', f"bookshelf_{bookshelf_id}_test.jsonl")
        
        # Process and save test books
        with open(test_file, 'w', encoding='utf-8') as f:
            for book in tqdm(test_books, desc="Processing test books"):
                content = self.process_single_book(book)
                if content:
                    # Create and write JSON line
                    json_line = json.dumps({"text": content}, ensure_ascii=False)
                    print(json_line, file=f)
                    
                    # Update book info
                    book_info = book.copy()
                    book_info.update({
                        'start_position': current_position,
                        'end_position': current_position + len(json_line),
                        'length': len(content)
                    })
                    successful_test_books.append(book_info)
                    test_positions.append((current_position, current_position + len(json_line)))
                    current_position += len(json_line) + 1  # +1 for newline
        
        # Process and save train books
        current_position = 0
        successful_train_books = []
        train_positions = []
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for book in tqdm(train_books, desc="Processing train books"):
                content = self.process_single_book(book)
                if content:
                    # Create and write JSON line
                    json_line = json.dumps({"text": content}, ensure_ascii=False)
                    print(json_line, file=f)
                    
                    # Update book info
                    book_info = book.copy()
                    book_info.update({
                        'start_position': current_position,
                        'end_position': current_position + len(json_line),
                        'length': len(content)
                    })
                    successful_train_books.append(book_info)
                    train_positions.append((current_position, current_position + len(json_line)))
                    current_position += len(json_line) + 1  # +1 for newline
        
        # Calculate file sizes
        train_size = os.path.getsize(train_file)
        test_size = os.path.getsize(test_file)
        
        # Save metadata with positions
        metadata = {
            'bookshelf_id': bookshelf_id,
            'train': {
                'num_books': len(successful_train_books),
                'books': successful_train_books,
                'total_chars': train_size,
                'format': 'jsonl',
                'file_path': os.path.relpath(train_file, self.output_dir)
            },
            'test': {
                'num_books': len(successful_test_books),
                'books': successful_test_books,
                'total_chars': test_size,
                'format': 'jsonl',
                'file_path': os.path.relpath(test_file, self.output_dir)
            }
        }
        
        metadata_file = os.path.join(self.output_dir, f"bookshelf_{bookshelf_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return train_file, test_file, metadata_file

    def process_single_book(self, book):
        """Process a single book and return filtered content."""
        content = self.download_book(book['id'])
        
        if content:
            # Remove Project Gutenberg header/footer
            lines = content.split('\n')
            start_idx = 0
            end_idx = len(lines)
            
            for i, line in enumerate(lines):
                if "*** START OF" in line:
                    start_idx = i + 1
                elif "*** END OF" in line:
                    end_idx = i
                    break
            
            filtered_content = '\n'.join(lines[start_idx:end_idx])
            
            if len(filtered_content.strip()) > 1000:
                return filtered_content
            else:
                self.logger.warning(f"Book {book['id']} filtered out: too short")
        
        return None

def main():
    parser = argparse.ArgumentParser(description='Scrape books from Project Gutenberg bookshelf')
    parser.add_argument('--bookshelf_id', type=str, required=True, help='Bookshelf ID (e.g., 57 for philosophy)')
    parser.add_argument('--test_size', type=int, default=2, help='Number of books for test set')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    
    args = parser.parse_args()
    
    scraper = GutenbergScraper(output_dir=args.output_dir, cache_dir=args.cache_dir)
    train_file, test_file, metadata_file = scraper.process_books(args.bookshelf_id, args.test_size)
    
    print(f"\nProcessing complete!")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"Metadata file: {metadata_file}")

if __name__ == "__main__":
    main()