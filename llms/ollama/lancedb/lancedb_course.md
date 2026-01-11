# LanceDB: Complete Learning Course

## Course Overview

This course will take you from zero to proficient in LanceDB, an embedded vector database designed for AI applications. You'll learn to build semantic search systems, implement RAG (Retrieval Augmented Generation), and create production-ready vector search solutions.

**Prerequisites:**

- Python programming (intermediate level)
- Basic understanding of databases
- Familiarity with NumPy/Pandas (helpful but not required)
- Basic machine learning concepts (helpful for embeddings section)

**Course Duration:** 8-10 hours (self-paced)

---

## Module 1: Understanding Vector Databases

### 1.1 What Are Vector Databases?

**Key Concepts:**

- Traditional databases vs. vector databases
- What are embeddings and why they matter
- Use cases: semantic search, recommendation systems, RAG, duplicate detection
- The similarity search problem

**Understanding Embeddings:**
Embeddings are dense numerical representations of data (text, images, audio) in high-dimensional space. Similar items are closer together in this space.

Example:

- "cat" → [0.2, 0.8, -0.3, ...]
- "kitten" → [0.19, 0.79, -0.28, ...] (close to "cat")
- "car" → [-0.5, 0.1, 0.9, ...] (far from "cat")

### 1.2 Why LanceDB?

**Advantages:**

- **Embedded**: No separate server to manage (like SQLite for vectors)
- **Serverless-friendly**: Works great in serverless environments
- **Multi-modal**: Text, images, and more
- **Disk-based**: Handles datasets larger than RAM
- **Fast**: Built on Lance format (columnar storage optimized for ML)
- **SQL support**: Familiar query interface
- **Python & JavaScript/TypeScript**: Multi-language support

**When to use LanceDB:**

- RAG applications
- Semantic search systems
- Recommendation engines
- Content deduplication
- Image similarity search
- Prototyping vector search features

---

## Module 2: Setup and First Steps

### 2.1 Installation

```bash
# Install LanceDB
pip install lancedb

# For working with embeddings, you'll also want:
pip install sentence-transformers  # Text embeddings
pip install pillow                  # Image processing
pip install openai                  # OpenAI embeddings (optional)
```

### 2.2 Your First LanceDB Database

```python
import lancedb

# Create/connect to a database
db = lancedb.connect("./my_database")

# Create a simple table
data = [
    {"id": 1, "vector": [0.1, 0.2, 0.3], "text": "Hello world"},
    {"id": 2, "vector": [0.2, 0.3, 0.4], "text": "LanceDB is awesome"},
    {"id": 3, "vector": [0.9, 0.8, 0.7], "text": "Vector databases rock"}
]

# Create table
table = db.create_table("my_first_table", data=data)

# Search for similar vectors
results = table.search([0.15, 0.25, 0.35]).limit(2).to_list()
print(results)
```

**Exercise 2.1:** Create a database with your own sample data and perform a basic search.

### 2.3 Understanding the Data Model

LanceDB tables are collections of records with:

- **Required:** Vector column (for similarity search)
- **Optional:** Metadata columns (any JSON-serializable data)

```python
# Schema example
{
    "vector": [float, float, float, ...],  # Embedding
    "text": str,                            # Original content
    "metadata": dict,                       # Any additional data
    "timestamp": datetime,
    "category": str
}
```

---

## Module 3: Core Operations (CRUD)

### 3.1 Creating Tables

```python
import lancedb
import pandas as pd

db = lancedb.connect("./data")

# Method 1: From list of dicts
data = [
    {"vector": [1, 2], "text": "First item"},
    {"vector": [3, 4], "text": "Second item"}
]
table = db.create_table("from_dicts", data=data)

# Method 2: From Pandas DataFrame
df = pd.DataFrame({
    "vector": [[1, 2], [3, 4]],
    "text": ["First", "Second"]
})
table = db.create_table("from_df", data=df)

# Method 3: Define schema explicitly
import pyarrow as pa

schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 128)),
    pa.field("text", pa.utf8()),
    pa.field("price", pa.float64())
])
table = db.create_table("with_schema", schema=schema)
```

### 3.2 Adding Data

```python
# Open existing table
table = db.open_table("my_table")

# Add single record
table.add([{"vector": [0.5, 0.6], "text": "New item"}])

# Add multiple records
new_data = [
    {"vector": [0.7, 0.8], "text": "Item 1"},
    {"vector": [0.9, 1.0], "text": "Item 2"}
]
table.add(new_data)

# Add from DataFrame
df = pd.DataFrame({...})
table.add(df)
```

### 3.3 Querying Data

```python
# Vector search (similarity search)
results = table.search([0.1, 0.2]) \
    .limit(5) \
    .to_pandas()

# With distance threshold
results = table.search([0.1, 0.2]) \
    .limit(10) \
    .distance_type("cosine") \
    .to_pandas()

# SQL-like filtering
results = table.search([0.1, 0.2]) \
    .where("price > 100") \
    .limit(5) \
    .to_list()

# Select specific columns
results = table.search([0.1, 0.2]) \
    .select(["text", "price"]) \
    .limit(5) \
    .to_pandas()
```

### 3.4 Updating and Deleting

```python
# Update records
table.update(where="id = 1", values={"text": "Updated text"})

# Delete records
table.delete("price < 50")

# Delete by IDs
table.delete("id IN (1, 2, 3)")
```

**Exercise 3.1:** Build a product database with vectors, names, prices, and categories. Practice CRUD operations.

---

## Module 4: Working with Embeddings

### 4.1 Text Embeddings with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer
import lancedb

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "All that glitters is not gold"
]

# Generate embeddings
embeddings = model.encode(documents)

# Store in LanceDB
db = lancedb.connect("./embeddings_db")
data = [
    {"vector": embedding.tolist(), "text": doc}
    for embedding, doc in zip(embeddings, documents)
]
table = db.create_table("quotes", data=data)

# Semantic search
query = "What does the phrase about a long trip mean?"
query_embedding = model.encode(query)

results = table.search(query_embedding.tolist()).limit(3).to_pandas()
print(results)
```

### 4.2 OpenAI Embeddings

```python
from openai import OpenAI
import lancedb

client = OpenAI()  # Requires OPENAI_API_KEY env variable

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

documents = [
    "Artificial intelligence is transforming industries",
    "Machine learning requires large datasets",
    "Neural networks mimic the human brain"
]

db = lancedb.connect("./openai_db")
data = [
    {"vector": get_embedding(doc), "text": doc}
    for doc in documents
]
table = db.create_table("ai_docs", data=data)

# Search
query = "How does AI impact business?"
query_vec = get_embedding(query)
results = table.search(query_vec).limit(2).to_list()
```

### 4.3 Embedding Function Integration

LanceDB supports automatic embedding generation:

```python
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# Get embedding function
model = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

# Define schema with automatic embeddings
class Document(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

db = lancedb.connect("./auto_embed_db")
table = db.create_table("docs", schema=Document)

# Add data - vectors generated automatically
table.add([
    {"text": "First document"},
    {"text": "Second document"}
])

# Search with automatic query embedding
results = table.search("query text").limit(5).to_list()
```

**Exercise 4.1:** Create a semantic search system for your own text collection (articles, notes, documentation).

---

## Module 5: Advanced Search Features

### 5.1 Distance Metrics

```python
# Cosine similarity (default, best for normalized vectors)
results = table.search(query_vec) \
    .distance_type("cosine") \
    .limit(5)

# L2 (Euclidean) distance
results = table.search(query_vec) \
    .distance_type("l2") \
    .limit(5)

# Dot product
results = table.search(query_vec) \
    .distance_type("dot") \
    .limit(5)
```

**When to use each:**

- **Cosine**: Text embeddings, when magnitude doesn't matter
- **L2**: When actual distance matters (geography, measurements)
- **Dot product**: When you want to favor larger magnitudes

### 5.2 Hybrid Search (Vector + Metadata)

```python
# Combine vector similarity with filters
results = table.search([0.1, 0.2, 0.3]) \
    .where("category = 'technology' AND price < 100") \
    .limit(10) \
    .to_pandas()

# Complex filters
results = table.search(query_vec) \
    .where("(category = 'tech' OR category = 'science') AND year >= 2020") \
    .limit(5)

# Filter then search
filtered_results = table.search(query_vec) \
    .where("rating > 4.5") \
    .limit(20)
```

### 5.3 Reranking

```python
# LanceDB doesn't have built-in reranking, but you can implement it:

def rerank_with_cross_encoder(results, query, cross_encoder_model):
    """Rerank search results using a cross-encoder model"""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(cross_encoder_model)

    # Get texts from results
    texts = [r['text'] for r in results]

    # Score each result against the query
    pairs = [[query, text] for text in texts]
    scores = model.predict(pairs)

    # Sort by new scores
    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [r for r, s in reranked]

# Usage
initial_results = table.search(query_vec).limit(100).to_list()
final_results = rerank_with_cross_encoder(initial_results, query_text, 'cross-encoder/ms-marco-MiniLM-L-6-v2')[:10]
```

### 5.4 Vector Indexes for Performance

```python
# Create IVF-PQ index for faster search
table.create_index(
    metric="cosine",
    num_partitions=256,  # Number of clusters
    num_sub_vectors=16   # For product quantization
)

# IVF (Inverted File Index)
table.create_index(
    metric="l2",
    index_type="IVF_PQ",
    num_partitions=256
)
```

**Index Types:**

- **IVF_PQ**: Inverted File with Product Quantization (good balance)
- **IVF_FLAT**: Faster, but uses more memory
- No index: Brute force (accurate but slow for large datasets)

**When to index:**

- Tables with > 10,000 vectors
- When query speed matters more than perfect accuracy
- Production systems

---

## Module 6: Building a RAG Application

### 6.1 RAG Architecture Overview

```text
User Query → Embedding → Vector Search → Context Retrieval → LLM → Response
```

### 6.2 Complete RAG Implementation

```python
import lancedb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class RAGSystem:
    def __init__(self, db_path="./rag_db"):
        self.db = lancedb.connect(db_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_client = OpenAI()

    def ingest_documents(self, documents, table_name="knowledge_base"):
        """Add documents to the vector database"""
        data = []
        for i, doc in enumerate(documents):
            embedding = self.embedding_model.encode(doc)
            data.append({
                "id": i,
                "text": doc,
                "vector": embedding.tolist()
            })

        # Create or replace table
        try:
            self.db.drop_table(table_name)
        except:
            pass

        self.table = self.db.create_table(table_name, data=data)
        print(f"Ingested {len(documents)} documents")

    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant documents"""
        query_embedding = self.embedding_model.encode(query)
        results = self.table.search(query_embedding.tolist()).limit(top_k).to_pandas()
        return results['text'].tolist()

    def generate_response(self, query, context):
        """Generate response using LLM with retrieved context"""
        context_text = "\n\n".join(context)

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer based on the context above:"}
        ]

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )

        return response.choices[0].message.content

    def query(self, question):
        """Complete RAG pipeline"""
        # 1. Retrieve relevant context
        context = self.retrieve_context(question)

        # 2. Generate response
        answer = self.generate_response(question, context)

        return {
            "answer": answer,
            "sources": context
        }

# Usage example
documents = [
    "Python is a high-level programming language known for its simplicity and readability.",
    "Machine learning is a subset of AI that enables systems to learn from data.",
    "LanceDB is an embedded vector database designed for AI applications.",
    "RAG stands for Retrieval Augmented Generation, combining search with LLMs."
]

rag = RAGSystem()
rag.ingest_documents(documents)

result = rag.query("What is RAG?")
print(f"Answer: {result['answer']}")
print(f"\nSources: {result['sources']}")
```

### 6.3 Advanced RAG: Chunking Strategy

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# Usage with large documents
def ingest_large_document(rag, document, doc_id, doc_title):
    """Ingest a large document with chunking"""
    chunks = chunk_text(document)

    data = []
    for i, chunk in enumerate(chunks):
        embedding = rag.embedding_model.encode(chunk)
        data.append({
            "vector": embedding.tolist(),
            "text": chunk,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "chunk_id": i,
            "total_chunks": len(chunks)
        })

    return data
```

**Exercise 6.1:** Build a RAG system for your personal documentation or notes.

---

## Module 7: Production Considerations

### 7.1 Performance Optimization

```python
# 1. Batch operations
embeddings = model.encode(documents, batch_size=32)

# 2. Use indexes for large tables
table.create_index(metric="cosine", num_partitions=256)

# 3. Limit result set
results = table.search(query_vec).limit(10)  # Don't retrieve more than needed

# 4. Select only needed columns
results = table.search(query_vec).select(["text", "id"]).limit(10)

# 5. Use appropriate distance metric
# Cosine is usually best for normalized embeddings
```

### 7.2 Error Handling

```python
import lancedb
from lancedb.exceptions import LanceDBError

try:
    db = lancedb.connect("./mydb")
    table = db.open_table("my_table")
except LanceDBError as e:
    print(f"Database error: {e}")
    # Handle error (create table, reconnect, etc.)

try:
    results = table.search(query_vec).limit(10).to_pandas()
except Exception as e:
    print(f"Search error: {e}")
    # Log error, return empty results, etc.
```

### 7.3 Monitoring and Logging

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoredRAG:
    def query(self, question):
        start_time = time.time()

        try:
            # Retrieve context
            retrieval_start = time.time()
            context = self.retrieve_context(question)
            retrieval_time = time.time() - retrieval_start

            # Generate response
            generation_start = time.time()
            answer = self.generate_response(question, context)
            generation_time = time.time() - generation_start

            total_time = time.time() - start_time

            logger.info(f"Query completed - Total: {total_time:.2f}s, "
                       f"Retrieval: {retrieval_time:.2f}s, "
                       f"Generation: {generation_time:.2f}s")

            return {"answer": answer, "sources": context}

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
```

### 7.4 Versioning and Backups

```python
# LanceDB stores data in a directory structure
# To backup: simply copy the directory

import shutil
from datetime import datetime

def backup_database(db_path, backup_dir="./backups"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{backup_dir}/backup_{timestamp}"
    shutil.copytree(db_path, backup_path)
    print(f"Backup created: {backup_path}")

# Restore: copy backup directory back
```

---

## Module 8: Real-World Projects

### Project 1: Document Search Engine

Build a semantic search engine for PDFs or markdown files.

**Requirements:**

- Ingest multiple documents
- Chunk long documents
- Semantic search across all documents
- Highlight relevant sections
- Source attribution

### Project 2: Chatbot with Memory

Create a chatbot that remembers conversation history.

**Requirements:**

- Store conversation history as embeddings
- Retrieve relevant past conversations
- Use context in responses
- Handle multi-turn conversations

### Project 3: Image Similarity Search

Build an image search engine using CLIP embeddings.

**Requirements:**

- Generate embeddings for images
- Text-to-image search
- Image-to-image similarity
- Filter by metadata (date, category, etc.)

```python
# Starter code for Image Search
from sentence_transformers import SentenceTransformer
from PIL import Image
import lancedb

model = SentenceTransformer('clip-ViT-B-32')

# Encode images
images = [Image.open(f"image{i}.jpg") for i in range(10)]
embeddings = model.encode(images)

# Store in LanceDB
db = lancedb.connect("./image_db")
data = [
    {"vector": emb.tolist(), "image_path": f"image{i}.jpg", "metadata": {}}
    for i, emb in enumerate(embeddings)
]
table = db.create_table("images", data=data)

# Text-to-image search
query = "a cat sitting on a couch"
query_embedding = model.encode(query)
results = table.search(query_embedding.tolist()).limit(5).to_pandas()
```

---

## Module 9: Integration Patterns

### 9.1 FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lancedb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Initialize
db = lancedb.connect("./api_db")
model = SentenceTransformer('all-MiniLM-L6-v2')

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

class Document(BaseModel):
    text: str
    metadata: dict = {}

@app.post("/add")
async def add_document(doc: Document):
    try:
        embedding = model.encode(doc.text)
        table = db.open_table("documents")
        table.add([{
            "vector": embedding.tolist(),
            "text": doc.text,
            "metadata": doc.metadata
        }])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query: SearchQuery):
    try:
        table = db.open_table("documents")
        query_embedding = model.encode(query.query)
        results = table.search(query_embedding.tolist()).limit(query.limit).to_pandas()
        return results.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 9.2 LangChain Integration

```python
from langchain.vectorstores import LanceDB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import lancedb

# Initialize
db = lancedb.connect("./langchain_db")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create LangChain vector store
vectorstore = LanceDB(
    connection=db,
    embedding=embeddings,
    table_name="documents"
)

# Add documents
documents = ["doc1", "doc2", "doc3"]
vectorstore.add_texts(documents)

# Similarity search
results = vectorstore.similarity_search("query", k=3)
```

---

## Module 10: Best Practices and Tips

### 10.1 Embedding Best Practices

1. **Choose the right model:**

   - Small datasets: Use smaller models (MiniLM)
   - High accuracy needs: Use larger models (mpnet, E5)
   - Multi-lingual: Use multi-lingual models
   - Domain-specific: Fine-tune or use domain-specific models

2. **Normalize embeddings:**

   - Most models output normalized vectors
   - If not, normalize before storing
   - Use cosine similarity for normalized vectors

3. **Consistent preprocessing:**
   - Apply same preprocessing to queries and documents
   - Lowercase, remove special chars (if model expects it)

### 10.2 Database Design

1. **Table organization:**

   - Separate tables for different content types
   - Don't mix embeddings of different dimensions
   - Use meaningful table names

2. **Metadata structure:**

   - Include source information
   - Add timestamps
   - Store original IDs for reference
   - Include filtering-friendly fields

3. **Schema example:**

```python
{
    "id": "unique_id",
    "vector": [float],
    "text": "original content",
    "title": "document title",
    "source": "source file/url",
    "created_at": datetime,
    "category": "tag",
    "metadata": {dict}  # Additional flexible fields
}
```

### 10.3 Search Quality Tips

1. **Query expansion:**

   - Rephrase user queries for better matches
   - Generate multiple query variations
   - Average embeddings of variations

2. **Result filtering:**

   - Set relevance thresholds
   - Filter by metadata first, then vector search
   - Combine with keyword search for hybrid approach

3. **Evaluation:**
   - Track search quality metrics
   - Collect user feedback
   - A/B test different approaches

### 10.4 Common Pitfalls

1. **Don't:**

   - Store very large documents without chunking
   - Ignore metadata - it's powerful for filtering
   - Use same embeddings for different languages
   - Forget to normalize when comparing distances

2. **Do:**
   - Chunk long documents (300-500 tokens optimal)
   - Create indexes for large datasets (>10k vectors)
   - Monitor search latency and accuracy
   - Version your embedding models

---

## 12

### Bulk Operations

```python
# Efficient bulk insert
def bulk_insert(table, documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embeddings = model.encode(batch)
        data = [
            {"vector": emb.tolist(), "text": doc}
            for emb, doc in zip(embeddings, batch)
        ]
        table.add(data)
```

### Database Management

```python
# List all tables
tables = db.table_names()

# Drop a table
db.drop_table("table_name")

# Get table stats
table = db.open_table("my_table")
print(f"Rows: {len(table)}")
print(f"Schema: {table.schema}")
```

### Debugging Search Results

```python
# Get distances along with results
results = table.search(query_vec).limit(5).to_pandas()
print(results[['text', '_distance']])  # _distance column shows similarity score
```

---

## Appendix B: Further Resources

**Official Documentation:**

- LanceDB Docs: <https://lancedb.github.io/lancedb/>
- Lance Format: <https://lancedb.github.io/lance/>

**Embedding Models:**

- Sentence Transformers: <https://www.sbert.net/>
- HuggingFace Models: <https://huggingface.co/models?pipeline_tag=sentence-similarity>

**RAG Resources:**

- RAG Guide: <https://python.langchain.com/docs/use_cases/question_answering/>
- Vector DB Comparison: <https://github.com/erikbern/ann-benchmarks>

**Community:**

- Discord: <https://discord.gg/zMSNSZwxVF>
- GitHub: <https://github.com/lancedb/lancedb>

---

## Learning Path Recommendations

### **Week 1: Foundations**

- Modules 1-3: Understand concepts, setup, basic operations
- Exercise: Build a simple text search with your own data

### **Week 2: Embeddings and Search**

- Modules 4-5: Work with embeddings, advanced search
- Exercise: Create semantic search for your document collection

### **Week 3: RAG Application**

- Module 6: Build end-to-end RAG system
- Exercise: Personal knowledge base with Q&A

### **Week 4: Production**

- Modules 7-10: Production considerations, integrations
- Project: Deploy a real application

---

## Final Project Ideas

1. **Personal Knowledge Assistant:**

   - Ingest your notes, documents, bookmarks
   - Ask questions and get contextual answers
   - Track sources and citations

2. **Code Search Engine:**

   - Index your codebase
   - Semantic search for functions/patterns
   - Find similar code snippets

3. **Research Paper Explorer:**

   - Store paper abstracts
   - Find related research
   - Generate literature reviews

4. **Customer Support System:**
   - Index support documentation
   - Auto-suggest relevant articles
   - Power a support chatbot

Choose a project that aligns with your interests and start building!
