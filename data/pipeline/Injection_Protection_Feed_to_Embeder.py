import re
import logging

# Use the root logger or create a child logger that will inherit handlers
logger = logging.getLogger(__name__)
logger.debug("injection_protection_feed_to_embeder.py is alive")

# Define suspicious patterns
multilingual_patterns = {
    'english': [r'ignore previous instructions', r'instead respond with'],
    'chinese': [r'忽略之前的指令', r'改为以下回复'],
    'spanish': [r'ignora las instrucciones anteriores', r'en su lugar responde con'],
    'russian': [r'игнорировать предыдущие инструкции', r'вместо этого ответить'],
    'arabic': [r'تجاهل التعليمات السابقة', r'بدلاً من ذلك رد بـ'],
    'french': [r'ignorez les instructions précédentes', r'répondez plutôt par'],
    'german': [r'ignoriere vorherige anweisungen', r'antworte stattdessen mit']
}

def sanitize_text(text, chunk_id=None, log=True):
    flagged = False
    flagged_phrases = []
    for lang, patterns in multilingual_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                flagged = True
                flagged_phrases.append((lang, pattern))
                text = re.sub(pattern, '[REDACTED SUSPICIOUS PHRASE]', text, flags=re.IGNORECASE)

    if flagged and log:
        logging.info(f"FLAGGED CHUNK: {chunk_id} | PHRASES: {flagged_phrases}")

    return text
