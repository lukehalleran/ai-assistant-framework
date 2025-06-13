# semantic_chunker.py

import os
import re
import json
from typing import List, Dict
from extract_wikipedia_articles import stream_extract_articles

class SemanticWikiChunker:
    """
    Chunks Wikipedia articles semantically for optimal FAISS embedding.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def parse_wiki_sections(self, wiki_text: str) -> List[Dict]:
        sections = []
        current_section = {"title": "Introduction", "content": [], "level": 0}

        for line in wiki_text.split('\n'):
            header_match = re.match(r'^(=+)\s*(.*?)\s*\1$', line)
            if header_match:
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content'])
                    sections.append(current_section)

                level = len(header_match.group(1))
                current_section = {
                    "title": header_match.group(2),
                    "content": [],
                    "level": level
                }
            else:
                if line.strip():
                    current_section['content'].append(line)

        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content'])
            sections.append(current_section)

        return sections

    def clean_wiki_markup(self, text: str) -> str:
        text = re.sub(r'\{\{[^}]+\}\}', '', text)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)
        text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
        text = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        words = text.split()
        chunks = []
        if not words:
            return chunks

        words_per_chunk = int(self.chunk_size / 1.5)
        words_per_overlap = int(self.chunk_overlap / 1.5)

        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + words_per_chunk
            chunk_text = ' '.join(words[start:end])

            if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_idx': chunk_idx,
                        'chunk_start': start,
                        'chunk_end': min(end, len(words))
                    }
                })
                chunk_idx += 1

            start = end - words_per_overlap
            if start >= len(words):
                break

        return chunks

    def process_article(self, article_dict: Dict) -> List[Dict]:
        # Use article_dict from stream_extract_articles
        article_data = {
            'title': article_dict.get('title', 'Unknown'),
            'page_id': article_dict.get('page_id', 'Unknown'),  # Optional, if you add page_id
            'raw_text': article_dict.get('text', '')
        }

        if not article_data['raw_text']:
            return []

        sections = self.parse_wiki_sections(article_data['raw_text'])
        all_chunks = []

        for section in sections:
            clean_text = self.clean_wiki_markup(section['content'])
            if not clean_text:
                continue

            section_metadata = {
                'title': article_data['title'],
                'page_id': article_data['page_id'],
                'section': section['title'],
                'section_level': section['level']
            }

            if self.estimate_tokens(clean_text) > self.chunk_size:
                chunks = self.chunk_text(clean_text, section_metadata)
                all_chunks.extend(chunks)
            else:
                all_chunks.append({
                    'text': clean_text,
                    'metadata': {
                        **section_metadata,
                        'chunk_idx': 0,
                        'chunk_start': 0,
                        'chunk_end': len(clean_text.split())
                    }
                })

        return all_chunks

    def save_chunks_jsonl(self, chunks: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    def stream_process_wikipedia(self, xml_path: str, output_dir: str,
                                 batch_size: int = 1000, max_articles: int = None):
        os.makedirs(output_dir, exist_ok=True)

        chunk_buffer = []
        file_idx = 0
        total_articles = 0
        total_chunks = 0

        for article_dict in stream_extract_articles(xml_path):
            if max_articles and total_articles >= max_articles:
                break

            chunks = self.process_article(article_dict)
            chunk_buffer.extend(chunks)
            total_articles += 1
            total_chunks += len(chunks)

            if total_articles % 10 == 0:
                print(f"[PROGRESS] {total_articles} articles → {total_chunks} chunks")

            if len(chunk_buffer) >= batch_size:
                output_path = os.path.join(output_dir, f'semantic_chunks_{file_idx:04}.jsonl')
                self.save_chunks_jsonl(chunk_buffer, output_path)
                print(f"[SAVED] {len(chunk_buffer)} chunks to {output_path}")
                chunk_buffer = []
                file_idx += 1

        if chunk_buffer:
            output_path = os.path.join(output_dir, f'semantic_chunks_{file_idx:04}.jsonl')
            self.save_chunks_jsonl(chunk_buffer, output_path)
            print(f"[SAVED] Final {len(chunk_buffer)} chunks")

        print(f"[COMPLETE] {total_articles} articles → {total_chunks} semantic chunks")
