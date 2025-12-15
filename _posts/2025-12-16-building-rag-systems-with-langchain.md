---
title: "Building Production-Ready RAG Systems with LangChain"
date: 2025-12-16 14:00:00 +1000
categories: [AI, Tutorial]
tags: [rag, langchain, python, vector-database, llm]
description: "A comprehensive guide to building Retrieval-Augmented Generation (RAG) systems using LangChain, covering architecture, implementation, and best practices."
image:
  path: /assets/img/posts/rag-architecture.png
  alt: "RAG System Architecture Diagram"
math: true
mermaid: true
---

## Introduction

Retrieval-Augmented Generation (RAG) has become the go-to pattern for building AI applications that need access to custom knowledge bases. In this post, we'll build a production-ready RAG system from scratch.

## What is RAG?

Large language models (LLMs) are trained on massive datasets, but that training has a cutoff date. Ask an LLM about something that happened last week, or about your company's internal documentation, and it's essentially guessing. This is the fundamental limitation RAG addresses.

**The core problem:** LLMs have impressive general knowledge but no awareness of your specific data. Fine-tuning is expensive, slow, and needs to be repeated whenever your data changes. You could stuff everything into the context window, but that hits token limits fast and gets expensive at scale.

**The RAG solution:** Instead of teaching the model everything upfront, you give it the ability to "look things up" at query time. When a user asks a question, the system first searches a knowledge base for relevant information, then hands that context to the LLM along with the question. The model generates its response grounded in actual source material rather than its parametric memory.

Think of it like the difference between a closed-book exam and an open-book exam. The LLM still needs to understand the material and reason about it, but it has reference material to work from.

### Why RAG Took Off

RAG became the dominant pattern for knowledge-intensive AI applications for a few reasons:

- **No retraining required** — Update your knowledge base anytime without touching model weights
- **Source attribution** — You can show users exactly where information came from
- **Reduced hallucination** — Grounding responses in retrieved text keeps the model honest
- **Cost effective** — Cheaper than fine-tuning, especially for frequently changing data

### The RAG Pipeline

At a high level, RAG systems work in two phases:

**Indexing (offline):** Your documents get chunked into smaller pieces, converted into numerical embeddings that capture semantic meaning, and stored in a vector database. This only happens once per document.

**Retrieval + Generation (runtime):** When a query comes in, it gets embedded using the same model, and the vector database returns the most semantically similar chunks. These chunks become context for the LLM, which generates the final answer.

```mermaid
flowchart LR
    A[User Query] --> B[Embedding Model]
    B --> C[Vector Search]
    C --> D[Retrieved Documents]
    D --> E[LLM + Context]
    E --> F[Response]
```
The quality of your RAG system depends heavily on that retrieval step. If you're pulling irrelevant chunks, even the best LLM will produce poor answers. That's why most of the engineering effort goes into chunking strategies, embedding selection, and retrieval optimization.

# Implementation

Building a RAG system requires three core components: a vector database, an embedding model, and an LLM. Here's a quick rundown of solid choices, then we'll get into the code.

## Choosing Your Stack

### Vector Database

- **Chroma** — Great for prototyping, runs locally, zero config
- **Pinecone** — Production-ready, fully managed, scales well
- **Qdrant** — Fast, open source, good middle ground

For production, Pinecone is hard to beat for reliability. For local dev and testing, Chroma gets you moving fast.

### Embedding Model

- **OpenAI text-embedding-3-small** — Cheap, good quality, 1536 dimensions
- **OpenAI text-embedding-3-large** — Better quality, 3072 dimensions
- **BGE / E5** — Free, self-hosted, competitive quality

OpenAI's large model hits a good balance of quality and ease of use.

### LLM

- **GPT-4o-mini** — Great quality-to-cost ratio
- **Llama 3.1 (8B/70B)** — Free, open source, run locally or via providers
- **Claude 3.5 Haiku** — Fast and cheap

As Open source models are catching up to proprietary models I've increasingly be using them more my favourite as of this date for RAG is Open-OSS.

---

## Prerequisites

Before we start building, ensure you have:

- Python 3.9+
- API keys for your chosen providers
- Basic understanding of embeddings and vector search

```bash
# Core dependencies
pip install langchain langchain-openai chromadb tiktoken

# Optional: for other providers
pip install pinecone-client      # Pinecone
pip install langchain-community  # BM25 and other retrievers
```

## Project Structure

```plaintext
rag-system/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── chain.py
│   └── main.py
├── data/
│   └── documents/
├── tests/
│   └── test_retriever.py
├── .env
└── requirements.txt
```

---

## Step 1: Document Loading and Chunking

Before embedding, documents need to be split into chunks. Chunk size significantly impacts retrieval quality.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory: str) -> list:
    """Load documents from a directory."""
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
```

> **Pro Tip:** 500-1000 tokens per chunk works well for most use cases. Too small and you lose context, too large and you dilute relevance.
{: .prompt-tip }

---

## Step 2: Creating the Vector Store

With Chroma (local development):

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(documents: list, persist_directory: str) -> Chroma:
    """Create and persist a vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store
```

With Pinecone (production):

```python
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def create_pinecone_store(documents: list, index_name: str) -> PineconeVectorStore:
    """Create a Pinecone vector store."""
    pc = Pinecone()  # Uses PINECONE_API_KEY env var
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    
    return vector_store
```

---

## Step 3: Building the RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(retriever):
    """Create a RAG chain with the given retriever."""
    
    template = """Answer the question based on the following context.
    If you cannot answer from the context, say so.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
```

---

## Step 4: Putting It All Together

```python
def main():
    # Load and process documents
    documents = load_documents("./data/documents")
    
    # Create vector store
    vector_store = create_vector_store(documents, "./chroma_db")
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create RAG chain
    chain = create_rag_chain(retriever)
    
    # Query the system
    response = chain.invoke("What is the main topic of the documents?")
    print(response)

if __name__ == "__main__":
    main()
```

---

## Advanced: Hybrid Search

Combine semantic search with keyword matching for better results:

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(documents, vector_store):
    """Create a hybrid retriever combining BM25 and vector search."""
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4
    
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    
    return ensemble_retriever
```

---

## Common Pitfalls

> **Warning:** Avoid these common mistakes:
{: .prompt-warning }

1. **Chunks too large** — Retrieved context contains irrelevant noise
2. **No overlap** — Important context split across chunk boundaries
3. **Poor prompting** — Not instructing the LLM to use the context properly
4. **Not testing retrieval** — Debug retrieval separately from generation

## Next Steps

In future posts, we'll cover:

- [ ] Adding reranking with Cohere or cross-encoders
- [ ] Implementing conversation memory
- [ ] Evaluating RAG performance with RAGAS
- [ ] Deploying with FastAPI

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Vector Database](https://www.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

*Have questions or suggestions? Reach out on [Twitter](https://twitter.com/willburnstech) or leave a comment below!*
