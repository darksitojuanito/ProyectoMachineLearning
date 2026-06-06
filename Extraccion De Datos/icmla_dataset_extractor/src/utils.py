import os
import logging
import time
import random
import json
import hashlib

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def get_logger(name):
    return logging.getLogger(name)

def random_delay(min_seconds=5, max_seconds=15):
    """Implementa una pausa aleatoria para no saturar los servidores."""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)

def get_cache_path(url, ext="json"):
    """Genera un nombre de archivo único para la URL dada."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{url_hash}.{ext}")

def save_to_cache(url, data, ext="json"):
    path = get_cache_path(url, ext)
    if ext == "json":
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)

def load_from_cache(url, ext="json"):
    path = get_cache_path(url, ext)
    if os.path.exists(path):
        if ext == "json":
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    return None
