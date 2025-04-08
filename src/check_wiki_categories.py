import gzip
import re
from collections import Counter

def peek_categories():
    with gzip.open('cache/wikipedia/enwiki-latest-categorylinks.sql.gz', 'rt', encoding='utf-8', errors='ignore') as f:
        categories = []
        for line in f:
            if 'philosophy' in line.lower():
                # Extract category names from INSERT statements
                matches = re.findall(r"'([^']*philosophy[^']*)'", line, re.IGNORECASE)
                categories.extend(matches)
                
        # Show most common philosophy-related categories
        counter = Counter(categories)
        print("\nMost common philosophy-related categories:")
        for cat, count in counter.most_common(30):
            print(f"{count:5d}: {cat}")

if __name__ == "__main__":
    peek_categories()