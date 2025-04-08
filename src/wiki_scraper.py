import os
import argparse
import requests
import logging
import json
import random
import bz2
from tqdm import tqdm
import subprocess
import glob
from typing import List, Tuple, Dict, Set
import shutil
import mysql.connector
from datetime import datetime

class WikipediaScraper:
    def __init__(self, output_dir="data", cache_dir="cache"):
        self.output_dir = os.path.join(output_dir, 'wikipedia')
        self.cache_dir = os.path.join(cache_dir, 'wikipedia')
        self.session = requests.Session()
        
        # URLs for both dumps we need
        self.dumps = {
            'pages': "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
            'categorylinks': "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-categorylinks.sql.gz"
        }
        
        # Create directories
        for dir_path in [self.output_dir, self.cache_dir,
                        os.path.join(self.output_dir, 'train'),
                        os.path.join(self.output_dir, 'test')]:
            os.makedirs(dir_path, exist_ok=True)

        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger('WikipediaScraper')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('wikipedia_scraper.log')
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        c_handler.setFormatter(fmt)
        f_handler.setFormatter(fmt)
        
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def download_dump(self, dump_type: str) -> str:
        """Download specified Wikipedia dump file if not present locally."""
        if dump_type not in self.dumps:
            raise ValueError(f"Unknown dump type: {dump_type}")

        url = self.dumps[dump_type]
        filename = url.split('/')[-1]
        file_path = os.path.join(self.cache_dir, filename)

        if os.path.exists(file_path):
            self.logger.info(f"{dump_type} dump file already exists: {file_path}")
            return file_path

        self.logger.info(f"Downloading {dump_type} dump from {url}")
        response = self.session.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)

        self.logger.info(f"Downloaded and saved to {file_path}")
        return file_path

    def verify_dump(self, dump_type: str) -> bool:
        """Verify if a dump file is complete and valid."""
        file_path = os.path.join(self.cache_dir, self.dumps[dump_type].split('/')[-1])
        
        if not os.path.exists(file_path):
            self.logger.info(f"{dump_type} dump file not found, needs downloading")
            return False
        
        try:
            if file_path.endswith('.bz2'):
                with bz2.open(file_path, 'rb') as f:
                    # Try to read the end of the file
                    f.seek(-8192, 2)
                    f.read(8192)
            elif file_path.endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rb') as f:
                    f.seek(-8192, 2)
                    f.read(8192)
                
            self.logger.info(f"{dump_type} dump file is valid: {file_path}")
            return True
        except Exception as e:
            self.logger.warning(f"{dump_type} dump file is incomplete or corrupted: {e}")
            return False

    def get_philosophy_page_ids(self, categorylinks_path: str) -> Set[int]:
        """
        Extract page IDs that belong to philosophy-related categories without using MySQL.
        """
        philosophy_categories = {
            # Main categories
            'Philosophy',
            'Philosophers',
            'Philosophical_concepts',
            'Philosophy_by_topic',
            
            # Major branches
            'Metaphysics',
            'Epistemology',
            'Logic',
            'Ethics',
            'Aesthetics',
            'Political_philosophy',
            'Philosophy_of_mind',
            'Philosophy_of_science',
            'Philosophy_of_language',
            'Moral_philosophy',
            'Social_philosophy',
            
            # Historical periods
            'Ancient_philosophy',
            'Medieval_philosophy',
            'Modern_philosophy',
            'Contemporary_philosophy',
            
            # Traditions
            'Western_philosophy',
            'Eastern_philosophy',
            'Islamic_philosophy',
            'Chinese_philosophy',
            'Indian_philosophy',
            
            # Schools and movements
            'Existentialism',
            'Phenomenology',
            'Critical_theory',
            'Analytic_philosophy',
            'Continental_philosophy',
            'Pragmatism',
            'Stoicism',
            'Empiricism',
            'Rationalism',
            
            # Specific areas
            'Philosophy_of_religion',
            'Philosophy_of_mathematics',
            'Philosophy_of_history',
            'Philosophy_of_education',
            'Philosophy_of_law',
            'Philosophy_of_technology',
            'Environmental_philosophy',
            'Feminist_philosophy',
            
            # Quality/importance categories
            'High-importance_Philosophy_articles',
            'Mid-importance_Philosophy_articles',
            'Philosophy_articles',
            'Philosophy_task_force_articles',
            
            # Additional concepts
            'Philosophical_methodology',
            'Philosophical_problems',
            'Philosophical_arguments',
            'Philosophical_theories',
            'Philosophy_literature'
        }
        
        page_ids = set()
        
        self.logger.info("Processing categorylinks file...")
        import gzip
        
        with gzip.open(categorylinks_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f):
                if 'INSERT INTO' in line:
                    values_start = line.find('VALUES ')
                    if values_start != -1:
                        values_part = line[values_start + 7:]
                        value_groups = values_part.split('),(')
                        for group in value_groups:
                            group = group.strip('();')
                            parts = group.split(',')
                            if len(parts) >= 2:
                                try:
                                    # First part is page_id
                                    page_id = int(parts[0].strip())
                                    # Second part is category name
                                    category = parts[1].strip("' ")
                                    
                                    # Check if category matches any in our set (case-insensitive)
                                    if any(phil_cat.lower() in category.lower() for phil_cat in philosophy_categories):
                                        # Exclude certain meta-categories
                                        if not any(exclude in category.lower() for exclude in [
                                            'wikipedian', 
                                            'redirect',
                                            'template',
                                            'category:',
                                            'disambiguation'
                                        ]):
                                            page_ids.add(page_id)
                                except (ValueError, IndexError):
                                    continue

        self.logger.info(f"Found {len(page_ids)} philosophy-related pages")
        return page_ids

    def extract_articles(self, pages_dump_path: str, page_ids: Set[int]) -> List[Dict]:
        """
        Extract articles from the pages dump that match our page IDs.
        """
        articles = []
        
        # Use WikiExtractor to get clean text
        import mwxml
        
        dump = mwxml.Dump.from_file(bz2.open(pages_dump_path))
        
        self.logger.info("Extracting matching articles...")
        for page in tqdm(dump):
            if page.id in page_ids:
                try:
                    # Get the latest revision
                    for revision in page:  # page is an iterator of revisions
                        latest_text = revision.text
                        break  # we only need the latest revision
                    
                    if latest_text:  # only add if we got some text
                        articles.append({
                            'id': page.id,
                            'title': page.title,
                            'text': latest_text
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing page {page.id}: {e}")
                    continue
        
        self.logger.info(f"Successfully extracted {len(articles)} articles")
        return articles

    def split_and_save_articles(self, articles: List[Dict], test_size: int) -> Tuple[str, str, str]:
        """Split articles into train/test sets and save them."""
        random.shuffle(articles)
        
        test_articles = articles[:test_size]
        train_articles = articles[test_size:]

        train_file = os.path.join(self.output_dir, 'train', "philosophy_train.jsonl")
        test_file = os.path.join(self.output_dir, 'test', "philosophy_test.jsonl")
        metadata_file = os.path.join(self.output_dir, "philosophy_metadata.json")

        # Write train file
        train_positions = self._write_articles_to_file(train_articles, train_file)
        
        # Write test file
        test_positions = self._write_articles_to_file(test_articles, test_file)

        # Create and save metadata
        metadata = {
            'topic': 'philosophy',
            'creation_date': datetime.now().isoformat(),
            'train': {
                'num_articles': len(train_articles),
                'articles': train_positions,
                'file_path': os.path.relpath(train_file, self.output_dir)
            },
            'test': {
                'num_articles': len(test_articles),
                'articles': test_positions,
                'file_path': os.path.relpath(test_file, self.output_dir)
            }
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return train_file, test_file, metadata_file

    def _write_articles_to_file(self, articles: List[Dict], filepath: str) -> List[Dict]:
        """Write articles to JSONL file and return position information."""
        positions = []
        current_pos = 0
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in articles:
                json_line = json.dumps({
                    "title": article['title'],
                    "text": article['text']
                }, ensure_ascii=False)
                f.write(json_line + '\n')
                
                position_info = {
                    'id': article['id'],
                    'title': article['title'],
                    'start_position': current_pos,
                    'end_position': current_pos + len(json_line),
                    'length': len(article['text'])
                }
                positions.append(position_info)
                current_pos += len(json_line) + 1

        return positions

    def process_wikipedia(self, test_size: int = 2) -> Tuple[str, str, str]:
        """Main processing function that handles the entire pipeline."""
        try:
            # Step 1: Get philosophy-related page IDs
            categorylinks_path = os.path.join(self.cache_dir, self.dumps['categorylinks'].split('/')[-1])
            page_ids = self.get_philosophy_page_ids(categorylinks_path)
            
            # Step 2: Extract matching articles
            pages_path = os.path.join(self.cache_dir, self.dumps['pages'].split('/')[-1])
            articles = self.extract_articles(pages_path, page_ids)
            
            # Step 3: Split and save articles
            return self.split_and_save_articles(articles, test_size)
            
        except Exception as e:
            self.logger.error(f"Error processing Wikipedia: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Download and process Wikipedia articles')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for processed files')
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Cache directory for downloaded files')
    parser.add_argument('--test_size', type=int, default=100,
                       help='Number of articles for test set')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['download_only', 'filter_only', 'download_and_filter'],
                       help='Operation mode: download_only, filter_only, or download_and_filter')

    args = parser.parse_args()
    scraper = WikipediaScraper(output_dir=args.output_dir, cache_dir=args.cache_dir)
    
    try:
        if args.mode == 'download_only':
            # Only download dumps
            for dump_type in ['pages', 'categorylinks']:
                scraper.download_dump(dump_type)
                
        elif args.mode == 'filter_only':
            # Process existing dumps
            train_file, test_file, metadata_file = scraper.process_wikipedia(test_size=args.test_size)
            print("\nProcessing complete!")
            print(f"Train file: {train_file}")
            print(f"Test file: {test_file}")
            print(f"Metadata file: {metadata_file}")
            
        elif args.mode == 'download_and_filter':
            # Do both
            for dump_type in ['pages', 'categorylinks']:
                scraper.download_dump(dump_type)
            train_file, test_file, metadata_file = scraper.process_wikipedia(test_size=args.test_size)
            print("\nProcessing complete!")
            print(f"Train file: {train_file}")
            print(f"Test file: {test_file}")
            print(f"Metadata file: {metadata_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()