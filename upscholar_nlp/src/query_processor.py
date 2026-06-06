import re
import unicodedata
from rapidfuzz import process, fuzz

SPANISH_TO_ENGLISH = {
    "aprendizaje profundo": "deep learning",
    "aprendizaje automatico": "machine learning",
    "redes neuronales": "neural networks",
    "red neuronal": "neural network",
    "reconocimiento facial": "face recognition",
    "biometria": "biometrics",
    "segmentacion": "segmentation",
    "segmentacion medica": "medical segmentation",
    "imagenes medicas": "medical images",
    "imagen medica": "medical image",
    "medico": "medical",
    "medica": "medical",
    "medic": "medical",
    "aprendizaje por refuerzo": "reinforcement learning",
    "refuerzo": "reinforcement",
    "series temporales": "time series",
    "serie temporal": "time series",
    "prediccion": "prediction",
    "pronostico": "forecasting",
    "grafos": "graphs",
    "grafo": "graph",
    "redes neuronales de grafos": "graph neural networks",
    "recomendacion": "recommendation",
    "clasificacion": "classification",
    "deteccion": "detection",
    "vision por computador": "computer vision",
    "procesamiento de lenguaje natural": "natural language processing",
    "lenguaje natural": "natural language",
    "salud": "health",
    "medicina": "medicine",
    "energia": "energy"
}

PROTECTED_TERMS = {"cnn", "gcn", "gan", "lstm", "rnn", "u-net", "bert", "yolo", "tf-idf"}

SPANISH_STOPWORDS = {"de", "la", "el", "los", "las", "en", "para", "por", "con", "un", "una", "unos", "unas", "sobre"}

PRIORITY_CORRECTIONS = {
    "learne": "learning",
    "machin": "machine",
    "netwroks": "networks",
    "segmetation": "segmentation",
    "recogntion": "recognition"
}

def remove_accents(input_str):
    return unicodedata.normalize('NFKD', input_str).encode('ASCII', 'ignore').decode('utf-8')

def build_vocabulary(df):
    vocab = set()
    try:
        import nltk
        from nltk.corpus import stopwords
        en_stops = set(stopwords.words('english'))
    except:
        en_stops = set()
        
    for col in ['Title', 'Keywords', 'Abstract']:
        if col in df.columns:
            for text in df[col].dropna():
                words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
                for w in words:
                    if w not in en_stops:
                        vocab.add(w)
    return list(vocab)

def remove_spanish_stopwords(query):
    words = query.split()
    filtered = [w for w in words if remove_accents(w.lower()) not in SPANISH_STOPWORDS]
    return " ".join(filtered)

def translate_spanish_terms(query):
    original_query = query
    # Normalize accents before replacing
    query_normalized = remove_accents(query)
    
    sorted_dict = sorted(SPANISH_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True)
    
    for es_term, en_term in sorted_dict:
        pattern = r'(?i)\b' + re.escape(es_term) + r'\b'
        query_normalized = re.sub(pattern, en_term, query_normalized)
        
    was_translated = (query_normalized.lower() != remove_accents(original_query).lower())
    return query_normalized, was_translated

def expand_wildcards(query, vocabulary):
    words = query.split()
    expanded_query = []
    expansions = []
    
    for word in words:
        if '*' in word or '?' in word:
            regex_pattern = '^' + re.escape(word).replace('\\*', '.*').replace('\\?', '.') + '$'
            compiled_regex = re.compile(regex_pattern, re.IGNORECASE)
            
            matches = [v for v in vocabulary if compiled_regex.match(v)]
            matches = matches[:10]
            
            if matches:
                expanded_query.extend(matches)
                expansions.append({"pattern": word, "matches": matches})
            else:
                expanded_query.append(word)
        else:
            expanded_query.append(word)
            
    return " ".join(expanded_query), expansions

def correct_typos(query, vocabulary):
    words = query.split()
    corrected_query = []
    corrections = []
    
    vocab_lower = {v.lower() for v in vocabulary}
    
    for word in words:
        word_lower = word.lower()
        
        if word_lower in PROTECTED_TERMS or '*' in word or '?' in word:
            corrected_query.append(word)
            continue
            
        # Priority Corrections Check (Overrides fuzzy matching)
        if word_lower in PRIORITY_CORRECTIONS:
            corrected = PRIORITY_CORRECTIONS[word_lower]
            corrected_query.append(corrected)
            corrections.append({"original": word, "corrected": corrected})
            continue
            
        if len(word) < 4 or word_lower in vocab_lower:
            corrected_query.append(word)
            continue
            
        match = process.extractOne(word_lower, vocabulary, scorer=fuzz.ratio)
        if match:
            best_match, score, _ = match
            if score >= 85:
                corrected_query.append(best_match)
                corrections.append({"original": word, "corrected": best_match})
            else:
                corrected_query.append(word)
        else:
            corrected_query.append(word)
            
    return " ".join(corrected_query), corrections

def process_query(query, vocabulary):
    original_query = query
    
    q = remove_spanish_stopwords(query)
    q, was_translated = translate_spanish_terms(q)
    q, expansions = expand_wildcards(q, vocabulary)
    q, corrections = correct_typos(q, vocabulary)
    
    q = re.sub(r'\s+', ' ', q).strip()
    
    return {
        "original_query": original_query,
        "processed_query": q,
        "was_translated": was_translated,
        "was_corrected": len(corrections) > 0,
        "used_wildcards": len(expansions) > 0,
        "corrections": corrections,
        "wildcard_expansions": expansions
    }
