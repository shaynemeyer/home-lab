# Module 4: Working with Embeddings - Deep Dive

## Table of Contents

1. [Understanding Embedding Models](#understanding-embedding-models)
2. [Sentence Transformers Deep Dive](#sentence-transformers-deep-dive)
3. [OpenAI Embeddings](#openai-embeddings)
4. [Other Embedding Providers](#other-embedding-providers)
5. [Automatic Embedding Generation](#automatic-embedding-generation)
6. [Choosing the Right Embedding Model](#choosing-the-right-embedding-model)
7. [Fine-Tuning Embeddings](#fine-tuning-embeddings)
8. [Best Practices and Optimization](#best-practices-and-optimization)
9. [Advanced Techniques](#advanced-techniques)
10. [Hands-On Projects](#hands-on-projects)

---

## Understanding Embedding Models

### What Makes a Good Embedding Model?

A quality embedding model should:

1. **Capture semantic meaning** - Similar concepts get similar vectors
2. **Preserve context** - "bank" (river) vs "bank" (financial) get different embeddings
3. **Be task-appropriate** - Optimized for your use case (search, classification, etc.)
4. **Consistent dimensionality** - Always output same-sized vectors
5. **Efficient** - Balance quality with speed/cost

### Embedding Model Architecture Evolution

```
2013: Word2Vec
├─ Each word = one vector
├─ No context understanding
└─ "bank" always gets same embedding

2018: BERT
├─ Contextual embeddings
├─ "bank" embedding depends on surrounding words
└─ Better semantic understanding

2019: Sentence-BERT
├─ Optimized for sentence similarity
├─ Faster than BERT for search
└─ Used in production today

2021: CLIP
├─ Images and text in same space
├─ "A cat photo" ≈ [cat image]
└─ Multi-modal capabilities

2023+: Modern models
├─ Longer context (512+ tokens)
├─ Multi-lingual
├─ Domain-specific variants
└─ More efficient architectures
```

### How Embedding Models Work

```python
# Simplified embedding process
def embedding_model(text):
    # 1. Tokenization
    tokens = tokenizer(text)  # ["Hello", "world"] → [101, 7592, 2088, 102]

    # 2. Neural Network Processing
    # Multiple transformer layers
    hidden_states = transformer_layers(tokens)

    # 3. Pooling
    # Convert variable-length tokens to fixed-size vector
    embedding = pooling(hidden_states)  # [768] dimensions

    # 4. Normalization (optional)
    embedding = normalize(embedding)  # Length = 1

    return embedding
```

### Embedding Quality Metrics

**How to evaluate embeddings:**

1. **Semantic Textual Similarity (STS)**

   - Benchmark: STS-B dataset
   - Measures if similar sentences get similar embeddings
   - Score: Spearman correlation (0-1, higher is better)

2. **Information Retrieval**

   - Benchmark: BEIR, MS MARCO
   - Measures search/retrieval quality
   - Metrics: nDCG@10, Recall@100

3. **Classification**
   - Use embeddings as features for classification
   - Accuracy on downstream tasks

```python
# Example: Test embedding quality
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentence pairs with human similarity ratings
pairs = [
    ("A man is eating food.", "A man is eating pasta.", 0.7),
    ("A plane is landing.", "An airplane is landing.", 0.9),
    ("A cat is playing.", "A dog is running.", 0.2),
]

# Compute model similarities
model_sims = []
human_sims = []

for sent1, sent2, human_rating in pairs:
    emb1 = model.encode(sent1)
    emb2 = model.encode(sent2)

    # Cosine similarity
    model_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    model_sims.append(model_sim)
    human_sims.append(human_rating)

# Correlation
correlation, _ = spearmanr(model_sims, human_sims)
print(f"Model quality (Spearman correlation): {correlation:.3f}")
```

---

## Sentence Transformers Deep Dive

### What is Sentence Transformers?

**Sentence Transformers** (SBERT) is a Python library that provides:

- State-of-the-art sentence embeddings
- 100+ pre-trained models
- Easy fine-tuning capabilities
- Optimized for semantic search and similarity tasks

### Installation and Setup

```bash
# Basic installation
pip install sentence-transformers

# With additional dependencies
pip install sentence-transformers[train]  # For fine-tuning

# GPU support (optional but recommended)
pip install sentence-transformers torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model (cached after first download)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
sentences = [
    "This is a sentence about artificial intelligence",
    "Machine learning is a subset of AI",
    "The weather is nice today"
]

# Single encoding
embedding = model.encode(sentences[0])
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Batch encoding (more efficient)
embeddings = model.encode(sentences, show_progress_bar=True)
print(f"All embeddings shape: {embeddings.shape}")  # (3, 384)

# Normalize embeddings (recommended for cosine similarity)
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
```

### Popular Models Compared

| Model                                   | Dimensions | Speed           | Quality         | Best For                 |
| --------------------------------------- | ---------- | --------------- | --------------- | ------------------------ |
| `all-MiniLM-L6-v2`                      | 384        | ⚡⚡⚡⚡⚡ Fast | ⭐⭐⭐⭐ Good   | General use, prototyping |
| `all-mpnet-base-v2`                     | 768        | ⚡⚡⚡ Medium   | ⭐⭐⭐⭐⭐ Best | Production, high quality |
| `all-MiniLM-L12-v2`                     | 384        | ⚡⚡⚡⚡ Fast   | ⭐⭐⭐⭐ Good   | Balanced speed/quality   |
| `multi-qa-mpnet-base-dot-v1`            | 768        | ⚡⚡⚡ Medium   | ⭐⭐⭐⭐⭐ Best | Question answering       |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384        | ⚡⚡⚡⚡ Fast   | ⭐⭐⭐⭐ Good   | Multi-lingual            |
| `msmarco-distilbert-base-v4`            | 768        | ⚡⚡⚡ Medium   | ⭐⭐⭐⭐⭐ Best | Document retrieval       |

### Choosing a Model

```python
def choose_model(use_case, priority='balanced'):
    """
    Helper to choose the right Sentence Transformer model

    Args:
        use_case: 'general', 'qa', 'multilingual', 'retrieval'
        priority: 'speed', 'quality', 'balanced'
    """
    models = {
        'general': {
            'speed': 'all-MiniLM-L6-v2',
            'balanced': 'all-MiniLM-L12-v2',
            'quality': 'all-mpnet-base-v2'
        },
        'qa': {
            'speed': 'multi-qa-MiniLM-L6-cos-v1',
            'balanced': 'multi-qa-mpnet-base-cos-v1',
            'quality': 'multi-qa-mpnet-base-dot-v1'
        },
        'multilingual': {
            'speed': 'paraphrase-multilingual-MiniLM-L12-v2',
            'balanced': 'paraphrase-multilingual-mpnet-base-v2',
            'quality': 'paraphrase-multilingual-mpnet-base-v2'
        },
        'retrieval': {
            'speed': 'msmarco-distilbert-base-v3',
            'balanced': 'msmarco-distilbert-base-v4',
            'quality': 'msmarco-MiniLM-L6-cos-v5'
        }
    }

    return models.get(use_case, {}).get(priority, 'all-MiniLM-L6-v2')

# Usage
model_name = choose_model('qa', priority='balanced')
model = SentenceTransformer(model_name)
print(f"Loaded: {model_name}")
```

### Advanced Encoding Options

```python
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Basic encoding
embeddings = model.encode(sentences)

# 2. Normalize embeddings
embeddings = model.encode(
    sentences,
    normalize_embeddings=True  # L2 normalization
)

# 3. Convert to numpy/torch
embeddings_np = model.encode(sentences, convert_to_numpy=True)
embeddings_torch = model.encode(sentences, convert_to_tensor=True)

# 4. Batch processing for large datasets
embeddings = model.encode(
    sentences,
    batch_size=32,  # Process 32 at a time
    show_progress_bar=True
)

# 5. GPU acceleration
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
embeddings = model.encode(sentences)

# 6. Precision control (for memory efficiency)
embeddings = model.encode(
    sentences,
    precision='float32'  # or 'int8', 'uint8', 'binary', 'ubinary'
)

# 7. Pooling strategies (advanced)
from sentence_transformers import models

word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,  # Mean pooling
    pooling_mode_cls_token=False,   # [CLS] token
    pooling_mode_max_tokens=False   # Max pooling
)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

### Semantic Search with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus of documents
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey."
]

# Encode corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Query
query = "A person is eating"
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute similarities
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

# Get top-k results
top_k = 5
top_results = torch.topk(cos_scores, k=top_k)

print(f"Query: {query}\n")
for score, idx in zip(top_results[0], top_results[1]):
    print(f"Score: {score:.4f} - {corpus[idx]}")
```

**Output:**

```text
Query: A person is eating

Score: 0.7082 - A man is eating food.
Score: 0.6819 - A man is eating a piece of bread.
Score: 0.2948 - The girl is carrying a baby.
Score: 0.2629 - A man is riding a horse.
Score: 0.1987 - A monkey is playing drums.
```

### Similarity Computation Utilities

```python
from sentence_transformers import util

# 1. Cosine similarity
cos_sim = util.cos_sim(embedding1, embedding2)

# 2. Pairwise cosine similarity
# Useful for comparing all embeddings to each other
all_similarities = util.cos_sim(embeddings, embeddings)

# 3. Dot product
dot_score = util.dot_score(embedding1, embedding2)

# 4. Euclidean distance
euclidean = util.euclidean_sim(embedding1, embedding2)

# 5. Manhattan distance
manhattan = util.manhattan_sim(embedding1, embedding2)

# 6. Semantic search (optimized)
hits = util.semantic_search(
    query_embeddings,
    corpus_embeddings,
    top_k=5,
    score_function=util.cos_sim  # or util.dot_score
)
```

### Handling Long Documents

```python
def encode_long_document(model, text, max_length=512, stride=256):
    """
    Encode long documents by chunking with overlap

    Args:
        model: SentenceTransformer model
        text: Long text to encode
        max_length: Maximum tokens per chunk
        stride: Overlap between chunks

    Returns:
        Average embedding of all chunks
    """
    # Tokenize
    tokens = model.tokenizer(text, add_special_tokens=False)['input_ids']

    # Create chunks with overlap
    chunks = []
    for i in range(0, len(tokens), max_length - stride):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = model.tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    # Encode all chunks
    chunk_embeddings = model.encode(chunks)

    # Average embeddings
    document_embedding = np.mean(chunk_embeddings, axis=0)

    # Normalize
    document_embedding = document_embedding / np.linalg.norm(document_embedding)

    return document_embedding

# Usage
long_text = "..." * 1000  # Very long text
embedding = encode_long_document(model, long_text)
```

### Multi-lingual Embeddings

```python
# Load multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Sentences in different languages
sentences = {
    'en': 'Hello, how are you?',
    'es': '¿Hola, cómo estás?',
    'fr': 'Bonjour, comment allez-vous?',
    'de': 'Hallo, wie geht es dir?',
    'zh': '你好，你好吗？'
}

# Encode all
embeddings = {lang: model.encode(text) for lang, text in sentences.items()}

# Compare cross-lingual similarity
print("Cross-lingual similarities:")
for lang1 in sentences:
    for lang2 in sentences:
        if lang1 < lang2:  # Avoid duplicates
            sim = util.cos_sim(embeddings[lang1], embeddings[lang2])[0][0]
            print(f"{lang1}-{lang2}: {sim:.3f}")
```

### Image Embeddings with CLIP

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Text embedding
text_emb = model.encode("A photo of a cat")

# Image embedding
img = Image.open('cat.jpg')
img_emb = model.encode(img)

# Text-to-image similarity
similarity = util.cos_sim(text_emb, img_emb)
print(f"Text-Image similarity: {similarity[0][0]:.3f}")

# Image-to-image similarity
img2 = Image.open('dog.jpg')
img2_emb = model.encode(img2)
similarity = util.cos_sim(img_emb, img2_emb)
print(f"Cat-Dog similarity: {similarity[0][0]:.3f}")
```

### Performance Optimization

```python
import time

def benchmark_encoding(model, sentences, batch_sizes=[1, 8, 32, 128]):
    """Benchmark different batch sizes"""

    results = {}

    for batch_size in batch_sizes:
        start = time.time()

        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False
        )

        duration = time.time() - start
        results[batch_size] = {
            'duration': duration,
            'sentences_per_sec': len(sentences) / duration
        }

    return results

# Test
sentences = ["This is a test sentence"] * 1000
model = SentenceTransformer('all-MiniLM-L6-v2')

results = benchmark_encoding(model, sentences)

print("Batch Size | Duration | Sentences/sec")
print("-" * 45)
for batch_size, metrics in results.items():
    print(f"{batch_size:10d} | {metrics['duration']:8.2f}s | {metrics['sentences_per_sec']:13.1f}")
```

**Expected output:**

```text
Batch Size | Duration | Sentences/sec
---------------------------------------------
         1 |    45.23s |          22.1
         8 |    12.34s |          81.0
        32 |     6.78s |         147.5
       128 |     5.91s |         169.2
```

### Memory Management

```python
# 1. Clear CUDA cache (if using GPU)
import torch
torch.cuda.empty_cache()

# 2. Use lower precision
embeddings = model.encode(
    sentences,
    precision='int8'  # Instead of 'float32'
)

# 3. Process in chunks for very large datasets
def encode_large_dataset(model, texts, chunk_size=10000):
    """Encode dataset in chunks to manage memory"""
    all_embeddings = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        chunk_embeddings = model.encode(chunk, show_progress_bar=True)
        all_embeddings.append(chunk_embeddings)

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.vstack(all_embeddings)

# 4. Use CPU for very large batches
model_cpu = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
```

---

## OpenAI Embeddings

### Overview

OpenAI provides high-quality embedding models via API:

**Current Models (as of 2024):**

- `text-embedding-3-small`: 1536 dimensions, fast and efficient
- `text-embedding-3-large`: 3072 dimensions, highest quality
- `text-embedding-ada-002`: 1536 dimensions (legacy, still popular)

**Pros:**

- State-of-the-art quality
- No local compute needed
- Reliable and scalable

**Cons:**

- Requires API key (costs money)
- Network latency
- Data leaves your environment
- Rate limits

### Setup

```bash
pip install openai
```

```python
import os
from openai import OpenAI

# Set API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Initialize client
client = OpenAI()
```

### OpenAI Embeddings - Basic Usage

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI"""
    text = text.replace("\n", " ")  # Replace newlines
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Single embedding
text = "This is a sample sentence"
embedding = get_embedding(text)

print(f"Embedding dimensions: {len(embedding)}")  # 1536
print(f"First 5 values: {embedding[:5]}")
```

### Batch Processing

```python
def get_embeddings_batch(texts, model="text-embedding-3-small", batch_size=100):
    """
    Process texts in batches to respect API limits

    OpenAI limits:
    - Max 8191 tokens per request
    - Recommended batch size: 100-1000
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Clean texts
        batch = [text.replace("\n", " ") for text in batch]

        # Get embeddings
        response = client.embeddings.create(
            input=batch,
            model=model
        )

        # Extract embeddings
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return all_embeddings

# Usage
texts = ["Text 1", "Text 2", ...] * 100  # 200 texts
embeddings = get_embeddings_batch(texts)
```

### Cost Optimization

```python
def estimate_cost(texts, model="text-embedding-3-small"):
    """
    Estimate cost for embedding generation

    Pricing (as of 2024):
    - text-embedding-3-small: $0.02 per 1M tokens
    - text-embedding-3-large: $0.13 per 1M tokens
    - text-embedding-ada-002: $0.10 per 1M tokens
    """
    import tiktoken

    # Get tokenizer for model
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    # Count tokens
    total_tokens = sum(len(encoding.encode(text)) for text in texts)

    # Pricing
    prices = {
        "text-embedding-3-small": 0.02 / 1_000_000,
        "text-embedding-3-large": 0.13 / 1_000_000,
        "text-embedding-ada-002": 0.10 / 1_000_000,
    }

    cost = total_tokens * prices[model]

    return {
        'total_tokens': total_tokens,
        'estimated_cost': cost,
        'cost_per_text': cost / len(texts)
    }

# Usage
texts = ["Sample text"] * 10000
estimate = estimate_cost(texts, model="text-embedding-3-small")

print(f"Total tokens: {estimate['total_tokens']:,}")
print(f"Estimated cost: ${estimate['estimated_cost']:.4f}")
print(f"Cost per text: ${estimate['cost_per_text']:.6f}")
```

### Dimensionality Reduction

```python
# New models support specifying output dimensions
def get_embedding_reduced(text, model="text-embedding-3-large", dimensions=512):
    """
    Get embedding with reduced dimensions

    Benefits:
    - Faster search
    - Less storage
    - Lower memory usage

    Trade-off:
    - Slightly lower quality
    """
    response = client.embeddings.create(
        input=[text],
        model=model,
        dimensions=dimensions  # Original: 3072, can reduce to 256-3072
    )
    return response.data[0].embedding

# Compare dimensions
text = "Sample text"
emb_full = get_embedding(text, model="text-embedding-3-large")
emb_reduced = get_embedding_reduced(text, model="text-embedding-3-large", dimensions=512)

print(f"Full dimensions: {len(emb_full)}")      # 3072
print(f"Reduced dimensions: {len(emb_reduced)}")  # 512
```

### Error Handling and Retry Logic

```python
import time
from openai import RateLimitError, APIError

def get_embedding_with_retry(text, model="text-embedding-3-small", max_retries=3):
    """Get embedding with exponential backoff retry"""

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=[text.replace("\n", " ")],
                model=model
            )
            return response.data[0].embedding

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff
            wait_time = 2 ** attempt
            print(f"Rate limit hit. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIError as e:
            print(f"API error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

# Usage
embedding = get_embedding_with_retry("Sample text")
```

### Caching Strategy

```python
import pickle
import hashlib
from pathlib import Path

class EmbeddingCache:
    """Cache OpenAI embeddings to disk"""

    def __init__(self, cache_dir="./embedding_cache", model="text-embedding-3-small"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model = model
        self.client = OpenAI()

    def _get_cache_key(self, text):
        """Generate cache key from text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{self.model}_{text_hash}.pkl"

    def get_embedding(self, text):
        """Get embedding with caching"""
        cache_file = self._get_cache_key(text)

        # Check cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Get from API
        response = self.client.embeddings.create(
            input=[text.replace("\n", " ")],
            model=self.model
        )
        embedding = response.data[0].embedding

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding

    def clear_cache(self):
        """Clear all cached embeddings"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

# Usage
cache = EmbeddingCache()

# First call: hits API
emb1 = cache.get_embedding("Sample text")

# Second call: uses cache (instant!)
emb2 = cache.get_embedding("Sample text")
```

### Integration with LanceDB

```python
import lancedb
from openai import OpenAI

class OpenAILanceDB:
    """LanceDB with OpenAI embeddings"""

    def __init__(self, db_path="./openai_db", model="text-embedding-3-small"):
        self.db = lancedb.connect(db_path)
        self.client = OpenAI()
        self.model = model
        self.embedding_dim = 1536  # For text-embedding-3-small

    def add_documents(self, documents, metadata=None):
        """Add documents with OpenAI embeddings"""
        data = []

        # Get embeddings in batch
        texts = [doc['text'] for doc in documents]
        embeddings = self._get_embeddings_batch(texts)

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            record = {
                'text': doc['text'],
                'vector': embedding
            }

            # Add metadata
            if metadata and i < len(metadata):
                record.update(metadata[i])

            data.append(record)

        # Create table
        self.table = self.db.create_table("documents", data=data, mode="overwrite")

        return len(data)

    def _get_embeddings_batch(self, texts, batch_size=100):
        """Get embeddings in batches"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [text.replace("\n", " ") for text in batch]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def search(self, query, top_k=5):
        """Search using OpenAI embeddings"""
        # Get query embedding
        query_embedding = self._get_embeddings_batch([query])[0]

        # Search
        results = self.table.search(query_embedding).limit(top_k).to_pandas()
        return results

# Usage
db = OpenAILanceDB()

documents = [
    {'text': 'Python programming language'},
    {'text': 'Machine learning algorithms'},
    {'text': 'Data science tools'}
]

db.add_documents(documents)
results = db.search("AI and programming")
print(results)
```

---

## Other Embedding Providers

### Cohere

```python
import cohere

co = cohere.Client('your-api-key')

# Get embeddings
response = co.embed(
    texts=['Hello world'],
    model='embed-english-v3.0',
    input_type='search_document'  # or 'search_query', 'classification', 'clustering'
)

embeddings = response.embeddings
```

### HuggingFace Inference API

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {YOUR_HF_TOKEN}"}

def query(texts):
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": texts}
    )
    return response.json()

embeddings = query(["Hello world", "Machine learning"])
```

### Voyage AI

```python
import voyageai

vo = voyageai.Client(api_key="your-api-key")

embeddings = vo.embed(
    ["Sample text 1", "Sample text 2"],
    model="voyage-2"
)
```

### Google Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(project='your-project')

def get_embedding(text):
    from vertexai.language_models import TextEmbeddingModel

    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings([text])
    return embeddings[0].values

embedding = get_embedding("Sample text")
```

---

## Automatic Embedding Generation

### LanceDB Embedding Functions

LanceDB supports automatic embedding generation using embedding functions:

```python
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# Get embedding function
embedder = get_registry().get("sentence-transformers").create(
    name="all-MiniLM-L6-v2"
)

# Define schema with automatic embeddings
class Document(LanceModel):
    text: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField()
    category: str

# Create database
db = lancedb.connect("./auto_embed_db")
table = db.create_table("docs", schema=Document, mode="overwrite")

# Add documents - vectors generated automatically!
table.add([
    {"text": "Python is great", "category": "programming"},
    {"text": "Machine learning rocks", "category": "ai"}
])

# Search - query embedding generated automatically!
results = table.search("coding").limit(5).to_list()
```

### Supported Embedding Functions

```python
from lancedb.embeddings import get_registry

# List all available
registry = get_registry()
print("Available embedding functions:")
for name in registry.list():
    print(f"- {name}")

# Common options:
# - sentence-transformers
# - openai
# - huggingface
# - cohere
# - ollama
```

### OpenAI Embedding Function

```python
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
import lancedb

# Create OpenAI embedder
openai_embedder = get_registry().get("openai").create(
    name="text-embedding-3-small",
    api_key="your-api-key"  # or set OPENAI_API_KEY env var
)

# Define schema
class Document(LanceModel):
    text: str = openai_embedder.SourceField()
    vector: Vector(openai_embedder.ndims()) = openai_embedder.VectorField()

# Use it
db = lancedb.connect("./openai_auto_db")
table = db.create_table("docs", schema=Document)

table.add([
    {"text": "First document"},
    {"text": "Second document"}
])

# Search with automatic query embedding
results = table.search("query text").limit(5).to_list()
```

### Custom Embedding Function

```python
from lancedb.embeddings import EmbeddingFunction, register
from typing import List
import numpy as np

@register("my-custom-embedder")
class CustomEmbedder(EmbeddingFunction):
    """Custom embedding function"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._ndims = self.model.get_sentence_embedding_dimension()

    def ndims(self):
        return self._ndims

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of documents"""
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query"""
        return self.model.encode(query, normalize_embeddings=True)

# Usage
embedder = CustomEmbedder()

class Document(LanceModel):
    text: str = embedder.SourceField()
    vector: Vector(embedder.ndims()) = embedder.VectorField()

db = lancedb.connect("./custom_embed_db")
table = db.create_table("docs", schema=Document)
```

### Multi-Modal Embedding Function

```python
from lancedb.embeddings import EmbeddingFunction, register
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

@register("clip-embedder")
class CLIPEmbedder(EmbeddingFunction):
    """CLIP embedder for text and images"""

    def __init__(self):
        self.model = SentenceTransformer('clip-ViT-B-32')
        self._ndims = 512  # CLIP dimension

    def ndims(self):
        return self._ndims

    def embed_documents(self, items: List):
        """Embed text or images"""
        embeddings = []
        for item in items:
            if isinstance(item, str):
                # Text
                emb = self.model.encode(item)
            elif isinstance(item, Image.Image):
                # Image
                emb = self.model.encode(item)
            else:
                raise ValueError(f"Unsupported type: {type(item)}")
            embeddings.append(emb)
        return embeddings

    def embed_query(self, query):
        """Embed query (text or image)"""
        return self.model.encode(query)
```

---

## Choosing the Right Embedding Model

### Decision Framework

```python
def recommend_embedding_model(
    use_case: str,
    budget: str,
    latency_requirement: str,
    data_privacy: bool,
    dataset_size: int
) -> dict:
    """
    Recommend embedding model based on requirements

    Args:
        use_case: 'search', 'qa', 'classification', 'clustering', 'multimodal'
        budget: 'free', 'low', 'medium', 'high'
        latency_requirement: 'realtime', 'fast', 'medium', 'batch'
        data_privacy: True if data must stay local
        dataset_size: Number of documents
    """

    # Filter by privacy
    if data_privacy:
        candidates = ['sentence-transformers', 'ollama', 'local-models']
    else:
        candidates = ['openai', 'cohere', 'sentence-transformers', 'huggingface']

    # Filter by budget
    if budget == 'free':
        candidates = [c for c in candidates if c in ['sentence-transformers', 'ollama']]

    # Choose model based on use case
    recommendations = {
        'search': {
            'model': 'all-mpnet-base-v2' if 'sentence-transformers' in candidates else 'text-embedding-3-small',
            'dimensions': 768 if 'sentence-transformers' in candidates else 1536,
            'reasoning': 'Optimized for semantic search and retrieval'
        },
        'qa': {
            'model': 'multi-qa-mpnet-base-dot-v1' if 'sentence-transformers' in candidates else 'text-embedding-3-small',
            'dimensions': 768,
            'reasoning': 'Trained specifically on question-answer pairs'
        },
        'classification': {
            'model': 'all-MiniLM-L6-v2',
            'dimensions': 384,
            'reasoning': 'Fast and efficient for classification tasks'
        },
        'clustering': {
            'model': 'all-mpnet-base-v2',
            'dimensions': 768,
            'reasoning': 'High-quality embeddings for grouping similar items'
        },
        'multimodal': {
            'model': 'clip-ViT-B-32',
            'dimensions': 512,
            'reasoning': 'Handles both text and images'
        }
    }

    recommendation = recommendations.get(use_case, recommendations['search'])

    # Add performance notes
    if dataset_size > 1_000_000:
        recommendation['note'] = 'Consider using IVF-PQ index for faster search'

    if latency_requirement == 'realtime':
        recommendation['note'] = 'Use smaller model (MiniLM) for lower latency'

    return recommendation

# Example usage
rec = recommend_embedding_model(
    use_case='search',
    budget='free',
    latency_requirement='fast',
    data_privacy=True,
    dataset_size=50000
)

print(f"Recommended: {rec['model']}")
print(f"Dimensions: {rec['dimensions']}")
print(f"Reasoning: {rec['reasoning']}")
```

### Comparison Benchmark

```python
import time
from sentence_transformers import SentenceTransformer
import numpy as np

def benchmark_models(texts, models_to_test):
    """Benchmark multiple embedding models"""

    results = {}

    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")

        model = SentenceTransformer(model_name)

        # Measure encoding time
        start = time.time()
        embeddings = model.encode(texts, show_progress_bar=False)
        duration = time.time() - start

        # Measure memory
        memory_mb = embeddings.nbytes / (1024 * 1024)

        # Test quality (simple check)
        test_pairs = [
            ("I love programming", "I enjoy coding", 0.8),  # Should be similar
            ("I love programming", "The weather is nice", 0.3),  # Should be different
        ]

        quality_scores = []
        for text1, text2, expected in test_pairs:
            emb1 = model.encode(text1)
            emb2 = model.encode(text2)
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            # Check if similarity matches expectation
            quality_scores.append(abs(similarity - expected) < 0.2)

        quality = np.mean(quality_scores) * 100

        results[model_name] = {
            'dimensions': model.get_sentence_embedding_dimension(),
            'time_per_1000': (duration / len(texts)) * 1000,
            'memory_mb': memory_mb,
            'quality_score': quality,
            'throughput': len(texts) / duration
        }

    return results

# Test models
test_texts = ["Sample text"] * 100
models = [
    'all-MiniLM-L6-v2',
    'all-MiniLM-L12-v2',
    'all-mpnet-base-v2',
]

results = benchmark_models(test_texts, models)

# Display results
print("\n" + "="*80)
print("Benchmark Results")
print("="*80)
print(f"{'Model':<30} {'Dims':<8} {'ms/1k':<10} {'Memory':<12} {'Quality':<10} {'Throughput':<12}")
print("-"*80)

for model_name, metrics in results.items():
    print(f"{model_name:<30} "
          f"{metrics['dimensions']:<8} "
          f"{metrics['time_per_1000']:<10.2f} "
          f"{metrics['memory_mb']:<12.2f} "
          f"{metrics['quality_score']:<10.1f}% "
          f"{metrics['throughput']:<12.1f} txt/s")
```

---

## Fine-Tuning Embeddings

### When to Fine-Tune

Fine-tune when:

- Domain-specific vocabulary (medical, legal, etc.)
- Your queries don't match pre-training data
- You have labeled data (query-document pairs)
- Quality improvements justify the effort

Don't fine-tune when:

- General domain (pre-trained works well)
- Small dataset (< 1000 examples)
- No labeled data
- Quick prototype

### Fine-Tuning with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Prepare training data
# Format: (text1, text2, similarity_score)
train_examples = [
    InputExample(texts=['Python programming', 'Coding in Python'], label=0.9),
    InputExample(texts=['Python programming', 'Java development'], label=0.3),
    InputExample(texts=['Machine learning', 'Artificial intelligence'], label=0.8),
    InputExample(texts=['Machine learning', 'Cooking recipes'], label=0.1),
    # Add hundreds or thousands more...
]

# 3. Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 4. Define loss
train_loss = losses.CosineSimilarityLoss(model)

# 5. Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path='./my-finetuned-model'
)

# 6. Use fine-tuned model
finetuned_model = SentenceTransformer('./my-finetuned-model')
```

### Fine-Tuning for Retrieval

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import models, evaluation

# Prepare data for retrieval task
# Format: (query, positive_document, negative_document)
train_examples = [
    InputExample(
        texts=[
            'How do I reset my password?',  # Query
            'Click forgot password on login page',  # Positive (relevant)
            'Our office hours are 9-5'  # Negative (not relevant)
        ]
    ),
    # More examples...
]

# Use Multiple Negatives Ranking Loss (best for retrieval)
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Evaluation
sentences1 = ['query 1', 'query 2']
sentences2 = ['doc 1', 'doc 2']
scores = [1.0, 0.8]  # Relevance scores

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1, sentences2, scores
)

# Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    evaluation_steps=500,
    output_path='./retrieval-model'
)
```

### Creating Training Data

```python
def create_training_data_from_clicks(user_queries, clicked_documents):
    """
    Create training data from user interactions

    Args:
        user_queries: List of search queries
        clicked_documents: List of documents users clicked for each query

    Returns:
        Training examples
    """
    examples = []

    for query, clicked_docs in zip(user_queries, clicked_documents):
        for doc in clicked_docs:
            # Positive pair (query, clicked document)
            # High label = more similar
            examples.append(
                InputExample(texts=[query, doc], label=0.9)
            )

    return examples

# Synthetic data generation
def generate_paraphrases(sentences, model_name='gpt-3.5-turbo'):
    """Generate paraphrases for training data augmentation"""
    from openai import OpenAI
    client = OpenAI()

    paraphrases = []
    for sentence in sentences:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Generate 3 different paraphrases of the given sentence."},
                {"role": "user", "content": sentence}
            ]
        )
        paraphrases.append(response.choices[0].message.content)

    return paraphrases
```

---

## Best Practices and Optimization

### 1. Text Preprocessing

```python
import re
from typing import List

class TextPreprocessor:
    """Preprocess text for embedding generation"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (optional, depends on use case)
        # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Lowercase (optional, most models handle this)
        # text = text.lower()

        return text.strip()

    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """Truncate to max tokens"""
        words = text.split()
        if len(words) > max_length:
            return ' '.join(words[:max_length])
        return text

    @staticmethod
    def prepare_for_embedding(texts: List[str]) -> List[str]:
        """Prepare batch of texts"""
        return [
            TextPreprocessor.clean_text(
                TextPreprocessor.truncate_text(text)
            )
            for text in texts
        ]

# Usage
preprocessor = TextPreprocessor()
cleaned = preprocessor.prepare_for_embedding([
    "  Text with   extra  spaces  ",
    "Very long text " * 1000
])
```

### 2. Batch Size Optimization

```python
def find_optimal_batch_size(model, sample_texts, max_batch_size=256):
    """Find optimal batch size for your hardware"""
    import torch

    results = {}

    for batch_size in [1, 8, 16, 32, 64, 128, 256]:
        if batch_size > max_batch_size:
            break

        try:
            # Test encoding
            start = time.time()
            embeddings = model.encode(
                sample_texts * (batch_size // len(sample_texts) + 1),
                batch_size=batch_size,
                show_progress_bar=False
            )
            duration = time.time() - start

            # Check memory if using GPU
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            else:
                memory_used = 0

            results[batch_size] = {
                'duration': duration,
                'throughput': len(embeddings) / duration,
                'memory_gb': memory_used
            }

        except RuntimeError as e:
            print(f"Batch size {batch_size} failed: {e}")
            break

    # Find optimal
    optimal = max(results.items(), key=lambda x: x[1]['throughput'])

    return optimal[0], results

# Usage
sample_texts = ["Sample text"] * 10
optimal_batch, results = find_optimal_batch_size(model, sample_texts)
print(f"Optimal batch size: {optimal_batch}")
```

### 3. Embedding Normalization

```python
import numpy as np

def normalize_embeddings(embeddings):
    """L2 normalization for cosine similarity"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Why normalize?
# 1. Makes cosine similarity = dot product (faster)
# 2. All vectors have length 1
# 3. Better for certain distance metrics

# Example
embeddings = model.encode(texts)
embeddings_normalized = normalize_embeddings(embeddings)

# Now cosine similarity is just dot product
similarities = np.dot(embeddings_normalized, embeddings_normalized.T)
```

### 4. Handling Long Documents

```python
def embed_long_document_hierarchical(model, document, chunk_size=512, stride=256):
    """
    Hierarchical embedding for very long documents

    1. Split into chunks
    2. Embed each chunk
    3. Use weighted average based on importance
    """
    from sentence_transformers import util

    # Split into sentences
    sentences = document.split('. ')

    # Create chunks
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += len(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Embed chunks
    chunk_embeddings = model.encode(chunks)

    # Weight by position (beginning and end are more important)
    weights = []
    for i in range(len(chunks)):
        if i == 0 or i == len(chunks) - 1:
            weights.append(2.0)  # Higher weight
        else:
            weights.append(1.0)

    weights = np.array(weights) / sum(weights)

    # Weighted average
    document_embedding = np.average(chunk_embeddings, axis=0, weights=weights)

    # Normalize
    document_embedding = document_embedding / np.linalg.norm(document_embedding)

    return document_embedding
```

### 5. Quality Monitoring

```python
class EmbeddingQualityMonitor:
    """Monitor embedding quality in production"""

    def __init__(self):
        self.metrics = []

    def check_quality(self, embeddings, texts):
        """Check for common issues"""
        issues = []

        # 1. Check for NaN or Inf
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            issues.append("NaN or Inf values detected")

        # 2. Check variance (low variance = poor embeddings)
        variance = np.var(embeddings)
        if variance < 0.01:
            issues.append(f"Low variance: {variance:.4f}")

        # 3. Check for duplicates (might indicate caching issues)
        unique_embeddings = np.unique(embeddings, axis=0)
        if len(unique_embeddings) < len(embeddings) * 0.9:
            duplicate_rate = 1 - (len(unique_embeddings) / len(embeddings))
            issues.append(f"High duplicate rate: {duplicate_rate:.2%}")

        # 4. Check norms (for normalized embeddings)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=0.01):
            issues.append("Embeddings not normalized")

        # 5. Sample similarity check
        if len(embeddings) >= 2:
            sim = np.dot(embeddings[0], embeddings[1]) / \
                  (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            if sim > 0.99:  # Very high similarity might be suspicious
                issues.append(f"Suspiciously high similarity: {sim:.4f}")

        self.metrics.append({
            'timestamp': time.time(),
            'num_embeddings': len(embeddings),
            'issues': issues,
            'variance': variance
        })

        return issues

# Usage
monitor = EmbeddingQualityMonitor()
embeddings = model.encode(texts)
issues = monitor.check_quality(embeddings, texts)

if issues:
    print("⚠️  Quality issues detected:")
    for issue in issues:
        print(f"  - {issue}")
```

---

## Advanced Techniques

### 1. Query Expansion

```python
def expand_query(query, model, top_k=3):
    """
    Expand query with similar terms for better retrieval

    Uses embedding similarity to find related concepts
    """
    # Vocabulary of terms (in practice, use your domain vocabulary)
    vocabulary = [
        "programming", "coding", "development", "software",
        "machine learning", "AI", "neural networks",
        "data science", "analytics", "statistics",
        # ... more terms
    ]

    # Embed query and vocabulary
    query_embedding = model.encode(query)
    vocab_embeddings = model.encode(vocabulary)

    # Find similar terms
    similarities = util.cos_sim(query_embedding, vocab_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:top_k]

    # Expand query
    expanded_terms = [vocabulary[i] for i in top_indices]
    expanded_query = query + " " + " ".join(expanded_terms)

    return expanded_query

# Usage
original = "How to code in Python?"
expanded = expand_query(original, model)
print(f"Original: {original}")
print(f"Expanded: {expanded}")
```

### 2. Hybrid Search (BM25 + Vector)

```python
from rank_bm25 import BM25Okapi

class HybridSearch:
    """Combine BM25 (keyword) and vector search"""

    def __init__(self, model):
        self.model = model
        self.documents = []
        self.embeddings = None
        self.bm25 = None

    def index(self, documents):
        """Index documents for hybrid search"""
        self.documents = documents

        # Vector embeddings
        self.embeddings = self.model.encode(documents)

        # BM25 index
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, top_k=10, alpha=0.5):
        """
        Hybrid search

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for vector search (1-alpha for BM25)
                  alpha=0: pure BM25
                  alpha=1: pure vector
                  alpha=0.5: balanced
        """
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)

        # Vector scores
        query_embedding = self.model.encode(query)
        vector_scores = util.cos_sim(query_embedding, self.embeddings)[0].numpy()

        # Combine scores
        hybrid_scores = alpha * vector_scores + (1 - alpha) * bm25_scores

        # Get top-k
        top_indices = hybrid_scores.argsort()[::-1][:top_k]

        results = [
            {
                'document': self.documents[i],
                'score': hybrid_scores[i],
                'vector_score': vector_scores[i],
                'bm25_score': bm25_scores[i]
            }
            for i in top_indices
        ]

        return results

# Usage
hybrid = HybridSearch(model)
hybrid.index(documents)

results = hybrid.search("machine learning algorithms", alpha=0.5)
for r in results[:3]:
    print(f"Score: {r['score']:.3f} - {r['document'][:50]}...")
```

### 3. Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

def rerank_results(query, results, cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Rerank results using cross-encoder

    Cross-encoder is slower but more accurate than bi-encoder
    Use for reranking top results
    """
    cross_encoder = CrossEncoder(cross_encoder_model)

    # Create pairs
    pairs = [[query, result['text']] for result in results]

    # Score pairs
    scores = cross_encoder.predict(pairs)

    # Sort by new scores
    reranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [{'text': r[0]['text'], 'score': r[1]} for r in reranked]

# Usage
# 1. Get initial results with bi-encoder (fast, less accurate)
initial_results = table.search(query_embedding).limit(100).to_list()

# 2. Rerank top results with cross-encoder (slow, more accurate)
final_results = rerank_results(query, initial_results[:20])
```

---

## Hands-On Projects

### Project 1: Build a Semantic FAQ System

```python
"""
Complete FAQ System with Multiple Embedding Models
"""

import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np

class SmartFAQ:
    def __init__(self, db_path="./faq_db"):
        self.db = lancedb.connect(db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.table = None

    def load_faqs(self, faqs):
        """
        Load FAQs

        Args:
            faqs: List of dicts with 'question', 'answer', 'category'
        """
        data = []
        for i, faq in enumerate(faqs):
            # Embed question
            embedding = self.model.encode(faq['question'])

            data.append({
                'id': i,
                'question': faq['question'],
                'answer': faq['answer'],
                'category': faq.get('category', 'general'),
                'vector': embedding.tolist()
            })

        self.table = self.db.create_table("faqs", data=data, mode="overwrite")
        print(f"✓ Loaded {len(faqs)} FAQs")

    def answer(self, user_question, top_k=3, category=None):
        """Answer user question"""
        # Embed question
        question_embedding = self.model.encode(user_question)

        # Search
        search = self.table.search(question_embedding).limit(top_k)

        # Filter by category if specified
        if category:
            search = search.where(f"category = '{category}'")

        results = search.to_pandas()

        # Get best match
        if len(results) > 0:
            best_match = results.iloc[0]

            return {
                'answer': best_match['answer'],
                'matched_question': best_match['question'],
                'confidence': 1 - best_match['_distance'],
                'alternatives': [
                    {'question': row['question'], 'answer': row['answer']}
                    for _, row in results.iloc[1:].iterrows()
                ]
            }

        return {'answer': "I don't know the answer to that question."}

    def add_faq(self, question, answer, category='general'):
        """Add new FAQ"""
        embedding = self.model.encode(question)

        new_id = len(self.table)
        self.table.add([{
            'id': new_id,
            'question': question,
            'answer': answer,
            'category': category,
            'vector': embedding.tolist()
        }])

        print(f"✓ Added new FAQ")

# Usage
faq_system = SmartFAQ()

faqs = [
    {
        'question': 'What are your business hours?',
        'answer': 'We are open Monday-Friday, 9 AM to 5 PM EST.',
        'category': 'general'
    },
    {
        'question': 'How do I reset my password?',
        'answer': 'Click the "Forgot Password" link on the login page and follow the instructions sent to your email.',
        'category': 'account'
    },
    # Add more...
]

faq_system.load_faqs(faqs)

# Test
result = faq_system.answer("When are you open?")
print(f"Q: When are you open?")
print(f"A: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Project 2: Document Similarity Finder

```python
"""
Find similar documents in a collection
"""

class SimilarityFinder:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.db = lancedb.connect("./similarity_db")

    def index_documents(self, documents):
        """Index documents with metadata"""
        data = []

        for i, doc in enumerate(documents):
            embedding = self.model.encode(doc['content'])

            data.append({
                'id': i,
                'title': doc.get('title', f'Document {i}'),
                'content': doc['content'],
                'author': doc.get('author', 'Unknown'),
                'date': doc.get('date', ''),
                'vector': embedding.tolist()
            })

        self.table = self.db.create_table("documents", data=data, mode="overwrite")
        print(f"✓ Indexed {len(documents)} documents")

    def find_similar(self, doc_id, top_k=5):
        """Find documents similar to given document"""
        # Get document embedding
        doc = self.table.search().where(f"id = {doc_id}").to_pandas()

        if doc.empty:
            return []

        doc_embedding = doc['vector'].iloc[0]

        # Find similar
        results = self.table.search(doc_embedding).limit(top_k + 1).to_pandas()

        # Exclude the document itself
        results = results[results['id'] != doc_id]

        return results.head(top_k).to_dict('records')

    def find_duplicates(self, threshold=0.95):
        """Find near-duplicate documents"""
        # Get all documents
        all_docs = self.table.search([0] * 768).limit(10000).to_pandas()

        duplicates = []

        for i, doc1 in all_docs.iterrows():
            # Search for similar documents
            results = self.table.search(doc1['vector']).limit(5).to_pandas()

            for _, doc2 in results.iterrows():
                if doc1['id'] != doc2['id']:
                    similarity = 1 - doc2['_distance']
                    if similarity > threshold:
                        duplicates.append({
                            'doc1': doc1['title'],
                            'doc2': doc2['title'],
                            'similarity': similarity
                        })

        return duplicates

# Usage
finder = SimilarityFinder()

documents = [
    {'title': 'Python Basics', 'content': 'Introduction to Python programming...', 'author': 'Alice'},
    {'title': 'Python 101', 'content': 'Getting started with Python...', 'author': 'Bob'},  # Similar!
    {'title': 'Java Guide', 'content': 'Java programming fundamentals...', 'author': 'Charlie'},
]

finder.index_documents(documents)

# Find similar documents
similar = finder.find_similar(doc_id=0, top_k=2)
print("Similar documents:")
for doc in similar:
    print(f"- {doc['title']} (similarity: {1 - doc['_distance']:.2%})")

# Find duplicates
duplicates = finder.find_duplicates(threshold=0.90)
print(f"\nFound {len(duplicates)} potential duplicates")
```

---

## Summary and Next Steps

### Key Takeaways

**Sentence Transformers:**

- ✓ Best for local, private deployments
- ✓ 100+ pre-trained models
- ✓ Fast and efficient
- ✓ Free to use

**OpenAI Embeddings:**

- ✓ State-of-the-art quality
- ✓ No local compute needed
- ✓ Simple API
- ✗ Costs money
- ✗ Data leaves your system

**Automatic Embedding:**

- ✓ Simplifies workflow
- ✓ Consistent embedding generation
- ✓ Built into LanceDB

### Decision Matrix

```text
Choose Sentence Transformers when:
├─ Data privacy is critical
├─ No budget for API calls
├─ Need offline capability
└─ Have local GPU resources

Choose OpenAI when:
├─ Need absolute best quality
├─ No local GPU available
├─ Small dataset (cost-effective)
└─ Rapid prototyping

Choose other providers when:
├─ Specific domain expertise needed
├─ Multi-lingual requirements
└─ Special use cases
```

### Practice Exercises

1. **Compare embedding models** on your dataset
2. **Build a hybrid search** system (BM25 + vectors)
3. **Fine-tune** a model for your domain
4. **Implement caching** to reduce API costs
5. **Create a monitoring dashboard** for embedding quality

### Further Resources

- Sentence Transformers Docs: <https://www.sbert.net>
- OpenAI Embeddings Guide: <https://platform.openai.com/docs/guides/embeddings>
- MTEB Leaderboard (model rankings): <https://huggingface.co/spaces/mteb/leaderboard>
- Fine-tuning Tutorial: <https://www.sbert.net/docs/training/overview.html>

---

**You now have deep knowledge of working with embeddings! Time to build something amazing!** 🚀
