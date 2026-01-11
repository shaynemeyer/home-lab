# Using LanceDB with Ollama: Complete Local RAG Tutorial

## Overview

This tutorial shows you how to build a **completely local RAG (Retrieval Augmented Generation) system** using:

- **LanceDB**: Local vector database for storing embeddings
- **Ollama**: Local LLM inference (no API keys, fully private)
- **Local embedding models**: Generate embeddings on your machine

**Benefits of this approach:**

- 🔒 **100% Private**: All data stays on your machine
- 💰 **Zero cost**: No API fees
- ⚡ **Fast**: No network latency
- 🌐 **Offline**: Works without internet

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Quick Start Example](#quick-start-example)
3. [Understanding the Components](#understanding-the-components)
4. [Building a Production RAG System](#building-a-production-rag-system)
5. [Advanced Patterns](#advanced-patterns)
6. [Real-World Examples](#real-world-examples)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Installation & Setup

### Step 1: Install Ollama

**macOS:**

```bash
# Download from https://ollama.ai or use Homebrew
brew install ollama

# Start Ollama service
ollama serve
```

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve
```

**Windows:**
Download from <https://ollama.ai> and run the installer.

### Step 2: Pull Models

```bash
# Pull an LLM (choose based on your hardware)
# Smaller models (8GB RAM+)
ollama pull llama3.2:3b

# Medium models (16GB RAM+)
ollama pull llama3.2:7b
ollama pull mistral:7b

# Larger models (32GB RAM+)
ollama pull llama3.1:70b

# Pull an embedding model
ollama pull nomic-embed-text
```

**Model recommendations:**

- **For responses:** `llama3.2:3b` (fast), `llama3.2:7b` (balanced), `mistral:7b` (quality)
- **For embeddings:** `nomic-embed-text` (best local embedding model)

### Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install lancedb ollama sentence-transformers numpy pandas
```

### Step 4: Verify Installation

```python
import ollama
import lancedb

# Test Ollama
response = ollama.chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': 'Say hello!'}]
)
print(response['message']['content'])

# Test LanceDB
db = lancedb.connect('./test_db')
print("LanceDB connected successfully!")
```

---

## Quick Start Example

Here's a minimal working example to get you started in 5 minutes:

```python
import ollama
import lancedb
from typing import List

# 1. Setup database
db = lancedb.connect("./my_rag_db")

# 2. Create sample documents
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning models require large amounts of training data.",
    "LanceDB is an embedded vector database for AI applications.",
    "Ollama allows you to run large language models locally on your computer.",
    "RAG combines retrieval systems with language models for better responses."
]

# 3. Generate embeddings using Ollama
def get_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama"""
    response = ollama.embeddings(
        model='nomic-embed-text',
        prompt=text
    )
    return response['embedding']

# 4. Store documents with embeddings
data = []
for doc in documents:
    embedding = get_embedding(doc)
    data.append({
        "text": doc,
        "vector": embedding
    })

table = db.create_table("documents", data=data, mode="overwrite")
print(f"✓ Indexed {len(documents)} documents")

# 5. RAG Query Function
def query_rag(question: str, top_k: int = 3) -> str:
    """Query the RAG system"""
    # Get question embedding
    question_embedding = get_embedding(question)

    # Search for relevant documents
    results = table.search(question_embedding).limit(top_k).to_list()

    # Build context from results
    context = "\n\n".join([r['text'] for r in results])

    # Generate response with Ollama
    prompt = f"""Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']

# 6. Try it out!
question = "How do I run language models on my computer?"
answer = query_rag(question)

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")
```

**Expected output:**

```text
✓ Indexed 5 documents

Question: How do I run language models on my computer?
Answer: According to the context, you can use Ollama to run large language models locally on your computer.
```

---

## Understanding the Components

### How Ollama Works

```python
import ollama

# Chat completion
response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain RAG in simple terms.'}
    ],
    stream=False  # Set to True for streaming responses
)
print(response['message']['content'])

# Generate embeddings
embedding = ollama.embeddings(
    model='nomic-embed-text',
    prompt='This is a sample text'
)
print(f"Embedding dimensions: {len(embedding['embedding'])}")
# Nomic-embed-text produces 768-dimensional vectors
```

### Why Nomic-Embed-Text?

Nomic-embed-text is specifically designed for:

- **Local operation**: Runs efficiently on consumer hardware
- **High quality**: Competitive with OpenAI's ada-002
- **Consistent dimensions**: Always 768-dimensional vectors
- **Fast**: Optimized for batch processing

### LanceDB Integration

```python
import lancedb

# Connect to database
db = lancedb.connect("./my_database")

# Create table with embeddings
table = db.create_table(
    "my_table",
    data=[
        {"vector": [0.1, 0.2, ...], "text": "content"},
    ]
)

# Search (cosine similarity by default)
results = table.search(query_vector).limit(5).to_list()
```

---

## Building a Production RAG System

### Complete RAG Class

```python
import ollama
import lancedb
from typing import List, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalRAG:
    """Production-ready RAG system using Ollama and LanceDB"""

    def __init__(
        self,
        db_path: str = "./rag_db",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:3b",
        table_name: str = "knowledge_base"
    ):
        """
        Initialize the RAG system

        Args:
            db_path: Path to LanceDB database
            embedding_model: Ollama embedding model name
            llm_model: Ollama LLM model name
            table_name: Name of the vector table
        """
        self.db = lancedb.connect(db_path)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.table_name = table_name
        self.table = None

        logger.info(f"Initialized LocalRAG with {llm_model}")

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def ingest_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 10
    ) -> None:
        """
        Ingest documents into the vector database

        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
            batch_size: Number of documents to process at once
        """
        logger.info(f"Ingesting {len(documents)} documents...")

        if metadata and len(metadata) != len(documents):
            raise ValueError("Metadata length must match documents length")

        data = []
        for i, doc in enumerate(documents):
            # Generate embedding
            embedding = self._get_embedding(doc)

            # Build record
            record = {
                "text": doc,
                "vector": embedding,
                "created_at": datetime.now().isoformat(),
                "doc_id": i
            }

            # Add metadata if provided
            if metadata:
                record.update(metadata[i])

            data.append(record)

            if (i + 1) % batch_size == 0:
                logger.info(f"Processed {i + 1}/{len(documents)} documents")

        # Create or replace table
        self.table = self.db.create_table(
            self.table_name,
            data=data,
            mode="overwrite"
        )

        logger.info(f"✓ Ingested {len(documents)} documents into {self.table_name}")

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add documents to existing table"""
        if self.table is None:
            self.table = self.db.open_table(self.table_name)

        data = []
        for i, doc in enumerate(documents):
            embedding = self._get_embedding(doc)
            record = {
                "text": doc,
                "vector": embedding,
                "created_at": datetime.now().isoformat()
            }
            if metadata:
                record.update(metadata[i])
            data.append(record)

        self.table.add(data)
        logger.info(f"✓ Added {len(documents)} documents")

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        filter_condition: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query
            top_k: Number of results to return
            filter_condition: SQL-like filter (e.g., "category = 'tech'")

        Returns:
            List of relevant documents with metadata
        """
        if self.table is None:
            self.table = self.db.open_table(self.table_name)

        # Generate query embedding
        query_embedding = self._get_embedding(query)

        # Search
        search = self.table.search(query_embedding).limit(top_k)

        # Apply filter if provided
        if filter_condition:
            search = search.where(filter_condition)

        results = search.to_list()

        logger.info(f"Retrieved {len(results)} documents for query")
        return results

    def generate_response(
        self,
        query: str,
        context: List[Dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using retrieved context

        Args:
            query: User question
            context: Retrieved documents
            system_prompt: Optional system prompt
            temperature: LLM temperature (0.0-1.0)

        Returns:
            Generated response
        """
        # Build context string
        context_text = "\n\n".join([
            f"[Document {i+1}]\n{doc['text']}"
            for i, doc in enumerate(context)
        ])

        # Build prompt
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer questions based only on the provided context. If the answer is not in the context, say so."

        user_prompt = f"""Context:
{context_text}

Question: {query}

Please provide a clear and concise answer based on the context above."""

        # Generate response
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': temperature,
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise

    def query(
        self,
        question: str,
        top_k: int = 3,
        filter_condition: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict:
        """
        Complete RAG query pipeline

        Args:
            question: User question
            top_k: Number of documents to retrieve
            filter_condition: Optional filter for retrieval
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optionally sources
        """
        logger.info(f"Processing query: {question}")

        # Retrieve relevant documents
        context = self.retrieve_context(question, top_k, filter_condition)

        # Generate response
        answer = self.generate_response(question, context)

        result = {"answer": answer}

        if return_sources:
            result["sources"] = [
                {
                    "text": doc["text"],
                    "metadata": {k: v for k, v in doc.items() if k not in ["text", "vector"]}
                }
                for doc in context
            ]

        return result

    def chat(
        self,
        messages: List[Dict],
        context_queries: Optional[List[str]] = None,
        top_k: int = 2
    ) -> str:
        """
        Chat with context from vector DB

        Args:
            messages: List of message dicts with 'role' and 'content'
            context_queries: Optional specific queries to retrieve context for
            top_k: Number of context documents per query

        Returns:
            Assistant response
        """
        # Get context if queries provided
        context_text = ""
        if context_queries:
            all_context = []
            for query in context_queries:
                results = self.retrieve_context(query, top_k)
                all_context.extend(results)

            context_text = "\n\n".join([doc['text'] for doc in all_context])
            context_text = f"\n\nRelevant context:\n{context_text}"

        # Build chat messages
        chat_messages = messages.copy()
        if context_text:
            # Add context to system message or create one
            if chat_messages and chat_messages[0]['role'] == 'system':
                chat_messages[0]['content'] += context_text
            else:
                chat_messages.insert(0, {
                    'role': 'system',
                    'content': f"You are a helpful assistant.{context_text}"
                })

        # Generate response
        response = ollama.chat(
            model=self.llm_model,
            messages=chat_messages
        )

        return response['message']['content']


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = LocalRAG(
        db_path="./my_rag_db",
        llm_model="llama3.2:3b"
    )

    # Ingest documents
    documents = [
        "Python is a versatile programming language used for web development, data science, and automation.",
        "Machine learning involves training models on data to make predictions or decisions.",
        "LanceDB is a vector database designed for storing and searching embeddings efficiently.",
        "Ollama enables running large language models locally without internet connectivity.",
        "RAG systems combine information retrieval with language generation for accurate responses."
    ]

    metadata = [
        {"category": "programming", "source": "manual"},
        {"category": "ai", "source": "manual"},
        {"category": "database", "source": "manual"},
        {"category": "ai", "source": "manual"},
        {"category": "ai", "source": "manual"}
    ]

    rag.ingest_documents(documents, metadata)

    # Query the system
    result = rag.query(
        "How can I run AI models locally?",
        top_k=2
    )

    print(f"\nQuestion: How can I run AI models locally?")
    print(f"Answer: {result['answer']}")
    print(f"\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['text'][:80]}...")
```

---

## Advanced Patterns

### 1. Streaming Responses

```python
def query_with_streaming(rag: LocalRAG, question: str, top_k: int = 3):
    """Query RAG with streaming response"""
    # Retrieve context
    context = rag.retrieve_context(question, top_k)
    context_text = "\n\n".join([doc['text'] for doc in context])

    # Build prompt
    prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""

    # Stream response
    print("Answer: ", end="", flush=True)
    stream = ollama.chat(
        model=rag.llm_model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )

    full_response = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end="", flush=True)
        full_response += content

    print()  # New line
    return full_response
```

### 2. Multi-Query Retrieval

```python
def multi_query_retrieve(rag: LocalRAG, question: str, num_variants: int = 3) -> List[Dict]:
    """Generate multiple query variants for better retrieval"""
    # Generate query variants
    prompt = f"""Generate {num_variants} different ways to ask the following question:

Question: {question}

Provide only the alternative questions, one per line."""

    response = ollama.chat(
        model=rag.llm_model,
        messages=[{'role': 'user', 'content': prompt}]
    )

    # Parse variants
    variants = [q.strip() for q in response['message']['content'].split('\n') if q.strip()]
    variants = [question] + variants[:num_variants]

    # Retrieve for each variant
    all_results = []
    seen_texts = set()

    for variant in variants:
        results = rag.retrieve_context(variant, top_k=3)
        for result in results:
            if result['text'] not in seen_texts:
                all_results.append(result)
                seen_texts.add(result['text'])

    return all_results[:5]  # Return top 5 unique results
```

### 3. Document Chunking for Long Texts

```python
def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split document into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

    return chunks


def ingest_long_documents(rag: LocalRAG, documents: List[Dict]):
    """
    Ingest long documents with chunking

    Args:
        documents: List of dicts with 'text', 'title', and other metadata
    """
    all_chunks = []
    all_metadata = []

    for doc in documents:
        chunks = chunk_document(doc['text'])

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                'title': doc.get('title', 'Untitled'),
                'chunk_id': i,
                'total_chunks': len(chunks),
                'source': doc.get('source', 'unknown')
            })

    rag.ingest_documents(all_chunks, all_metadata)
```

### 4. Conversational RAG with Memory

```python
class ConversationalRAG(LocalRAG):
    """RAG with conversation memory"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []

    def query_conversational(self, question: str, top_k: int = 3) -> Dict:
        """Query with conversation context"""
        # Retrieve relevant documents
        context = self.retrieve_context(question, top_k)
        context_text = "\n\n".join([doc['text'] for doc in context])

        # Build messages with history
        messages = [
            {'role': 'system', 'content': f'You are a helpful assistant. Use this context: {context_text}'}
        ]
        messages.extend(self.conversation_history)
        messages.append({'role': 'user', 'content': question})

        # Generate response
        response = ollama.chat(
            model=self.llm_model,
            messages=messages
        )

        answer = response['message']['content']

        # Update history
        self.conversation_history.append({'role': 'user', 'content': question})
        self.conversation_history.append({'role': 'assistant', 'content': answer})

        # Keep only last 10 exchanges
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return {"answer": answer, "sources": context}

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Usage
conv_rag = ConversationalRAG()
conv_rag.ingest_documents(documents)

result1 = conv_rag.query_conversational("What is Python?")
print(result1['answer'])

result2 = conv_rag.query_conversational("What are its main uses?")  # Refers to Python
print(result2['answer'])
```

---

## Real-World Examples

### Example 1: Personal Knowledge Base

```python
import os
import glob

def build_personal_kb():
    """Build a knowledge base from your personal documents"""
    rag = LocalRAG(db_path="./personal_kb", llm_model="llama3.2:7b")

    # Load documents from directory
    documents = []
    metadata = []

    for filepath in glob.glob("./my_docs/**/*.txt", recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Chunk if needed
        if len(content.split()) > 500:
            chunks = chunk_document(content)
            documents.extend(chunks)
            metadata.extend([
                {'filename': os.path.basename(filepath), 'filepath': filepath}
                for _ in chunks
            ])
        else:
            documents.append(content)
            metadata.append({'filename': os.path.basename(filepath), 'filepath': filepath})

    # Ingest
    rag.ingest_documents(documents, metadata)
    print(f"Indexed {len(documents)} document chunks")

    return rag


# Usage
kb = build_personal_kb()

while True:
    question = input("\nAsk a question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break

    result = kb.query(question)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['metadata']['filename']}")
```

### Example 2: Code Documentation Assistant

```python
def build_code_docs_assistant(repo_path: str):
    """Build assistant for code documentation"""
    rag = LocalRAG(
        db_path="./code_docs_db",
        llm_model="llama3.2:7b"
    )

    documents = []
    metadata = []

    # Find all code files
    extensions = ['*.py', '*.js', '*.ts', '*.go', '*.rs']
    for ext in extensions:
        for filepath in glob.glob(f"{repo_path}/**/{ext}", recursive=True):
            # Skip node_modules, venv, etc.
            if any(skip in filepath for skip in ['node_modules', 'venv', '__pycache__', '.git']):
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Only index files with docstrings/comments
                if '"""' in content or '///' in content or '/*' in content:
                    documents.append(content)
                    metadata.append({
                        'filepath': filepath.replace(repo_path, ''),
                        'language': ext[2:]  # Remove *.
                    })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    rag.ingest_documents(documents, metadata)
    return rag


# Usage
docs_assistant = build_code_docs_assistant('./my_project')

result = docs_assistant.query(
    "How do I handle authentication in this codebase?",
    top_k=5
)
print(result['answer'])
```

### Example 3: Research Paper Q&A

```python
def ingest_pdf_documents(rag: LocalRAG, pdf_directory: str):
    """Ingest PDF documents (requires PyPDF2)"""
    try:
        import PyPDF2
    except ImportError:
        print("Install PyPDF2: pip install PyPDF2")
        return

    documents = []
    metadata = []

    for pdf_file in glob.glob(f"{pdf_directory}/*.pdf"):
        with open(pdf_file, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)

            # Extract text from all pages
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

            # Chunk the document
            chunks = chunk_document(text, chunk_size=400)

            documents.extend(chunks)
            metadata.extend([
                {
                    'filename': os.path.basename(pdf_file),
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
                for i in range(len(chunks))
            ])

    rag.ingest_documents(documents, metadata)
    print(f"Ingested {len(documents)} chunks from {len(set(m['filename'] for m in metadata))} PDFs")


# Usage
research_rag = LocalRAG(db_path="./research_papers")
ingest_pdf_documents(research_rag, "./papers")

result = research_rag.query(
    "What are the main findings about transformer architectures?",
    top_k=5
)
print(result['answer'])
```

---

## Performance Optimization

### 1. Batch Processing

```python
def ingest_large_dataset(rag: LocalRAG, documents: List[str], batch_size: int = 50):
    """Efficiently ingest large document collections"""
    from tqdm import tqdm

    all_data = []

    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch = documents[i:i + batch_size]

        # Process batch
        for doc in batch:
            embedding = rag._get_embedding(doc)
            all_data.append({
                "text": doc,
                "vector": embedding
            })

    # Single insert
    rag.table = rag.db.create_table(
        rag.table_name,
        data=all_data,
        mode="overwrite"
    )

    print(f"✓ Ingested {len(all_data)} documents")
```

### 2. Caching Embeddings

```python
import pickle
from pathlib import Path

class CachedRAG(LocalRAG):
    """RAG with embedding cache"""

    def __init__(self, *args, cache_path: str = "./embedding_cache.pkl", **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if self.cache_path.exists():
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    def _get_embedding(self, text: str) -> List[float]:
        # Check cache
        if text in self.cache:
            return self.cache[text]

        # Generate and cache
        embedding = super()._get_embedding(text)
        self.cache[text] = embedding

        # Save cache periodically
        if len(self.cache) % 100 == 0:
            self._save_cache()

        return embedding
```

### 3. Indexing for Large Datasets

```python
def create_optimized_table(db: lancedb.LanceDBConnection, data: List[Dict], table_name: str):
    """Create table with optimized index"""
    # Create table
    table = db.create_table(table_name, data=data, mode="overwrite")

    # Create IVF-PQ index for faster search
    if len(data) > 10000:
        table.create_index(
            metric="cosine",
            num_partitions=256,
            num_sub_vectors=16
        )
        print("✓ Created IVF-PQ index for fast search")

    return table
```

### 4. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_embed(rag: LocalRAG, documents: List[str], max_workers: int = 4) -> List[Dict]:
    """Generate embeddings in parallel"""
    def process_doc(doc):
        embedding = rag._get_embedding(doc)
        return {"text": doc, "vector": embedding}

    data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_doc, doc) for doc in documents]

        for future in as_completed(futures):
            data.append(future.result())

    return data
```

---

## Troubleshooting

### Common Issues

#### **1. Ollama Connection Error**

```python
# Problem: Cannot connect to Ollama
# Solution: Ensure Ollama is running

import subprocess
subprocess.run(['ollama', 'serve'], check=False)
```

#### **2. Model Not Found**

```python
# Problem: Model not available
# Solution: Pull the model first

import subprocess
subprocess.run(['ollama', 'pull', 'nomic-embed-text'], check=True)
subprocess.run(['ollama', 'pull', 'llama3.2:3b'], check=True)
```

#### **3. Out of Memory**

```python
# Problem: Model too large for RAM
# Solution: Use smaller model or quantized version

# Instead of llama3.1:70b, use:
# - llama3.2:3b (smaller)
# - llama3.2:7b (medium)
# - mistral:7b (efficient)
```

#### **4. Slow Embedding Generation**

```python
# Problem: Embeddings take too long
# Solution: Use caching and batch processing

# Use CachedRAG class (shown above)
# Process in batches
```

#### **5. Poor Search Results**

```python
# Problem: Retrieved documents not relevant
# Solution: Adjust search parameters

# Try more results
results = rag.retrieve_context(query, top_k=10)

# Try different chunking
chunks = chunk_document(text, chunk_size=300, overlap=75)

# Use metadata filters
results = rag.retrieve_context(query, filter_condition="category = 'relevant_type'")
```

### Debugging Tips

```python
# Check embedding dimensions
embedding = rag._get_embedding("test")
print(f"Embedding dimensions: {len(embedding)}")

# Inspect search results with distances
results = rag.retrieve_context("query", top_k=5)
for r in results:
    print(f"Distance: {r.get('_distance', 'N/A')}")
    print(f"Text: {r['text'][:100]}...")

# Test LLM directly
response = ollama.chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
print(response['message']['content'])
```

---

## Complete Working Example

Here's a full script you can run right away:

```python
#!/usr/bin/env python3
"""
Complete Local RAG System with Ollama and LanceDB
Run: python local_rag.py
"""

import ollama
import lancedb
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAG:
    def __init__(self, db_path: str = "./simple_rag"):
        self.db = lancedb.connect(db_path)
        self.embedding_model = "nomic-embed-text"
        self.llm_model = "llama3.2:3b"
        self.table_name = "docs"

    def embed(self, text: str) -> List[float]:
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response['embedding']

    def ingest(self, documents: List[str]):
        data = [{"text": doc, "vector": self.embed(doc)} for doc in documents]
        self.table = self.db.create_table(self.table_name, data=data, mode="overwrite")
        logger.info(f"Ingested {len(documents)} documents")

    def query(self, question: str, top_k: int = 3) -> str:
        # Search
        query_vec = self.embed(question)
        results = self.table.search(query_vec).limit(top_k).to_list()

        # Build context
        context = "\n\n".join([r['text'] for r in results])

        # Generate answer
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )

        return response['message']['content']


def main():
    # Initialize
    rag = SimpleRAG()

    # Sample knowledge base
    docs = [
        "LanceDB is an embedded vector database for AI applications.",
        "Ollama lets you run large language models locally.",
        "RAG combines retrieval with generation for better answers.",
        "Python is great for building AI applications.",
        "Vector embeddings represent text as numerical arrays."
    ]

    # Ingest
    rag.ingest(docs)

    # Interactive loop
    print("\n🤖 Local RAG System Ready!")
    print("Ask questions (type 'quit' to exit)\n")

    while True:
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break

        answer = rag.query(question)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
```

Save this as `local_rag.py` and run:

```bash
python local_rag.py
```

---

## Next Steps

1. **Expand your knowledge base**: Add more documents relevant to your domain
2. **Tune parameters**: Experiment with chunk sizes, top_k values, and models
3. **Add features**: Implement filters, multi-modal search, or conversation memory
4. **Build a UI**: Create a web interface with Streamlit or Gradio
5. **Deploy**: Package as a standalone application

## Additional Resources

- [**Ollama Documentation**](https://github.com/ollama/ollama)
- [**LanceDB Documentation**](https://lancedb.github.io/lancedb/)
- [**Nomic Embed**](https://huggingface.co/nomic-ai/nomic-embed-text-v1)

---

**You now have everything needed to build powerful local RAG systems!** 🚀
