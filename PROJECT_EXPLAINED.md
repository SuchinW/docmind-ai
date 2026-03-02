# DocMind AI - In-Depth Project Explanation (Interview Ready)

---

## 1. What Problem Does This Project Solve?

LLMs like GPT-4 and Claude are powerful, but they have two critical limitations:

1. **Knowledge cutoff** — They don't know about your private/custom documents.
2. **Hallucination** — They sometimes make up answers confidently.

DocMind AI solves both by using **RAG (Retrieval-Augmented Generation)**. Instead of relying on the LLM's training data, we:
- Feed it your actual documents as context
- Force it to answer ONLY from that context
- Make it cite which source it used

**Interview Q: "Why not just paste the whole document into the LLM prompt?"**
> Because LLMs have **token limits** (e.g., GPT-4o-mini has ~128K tokens). A 500-page PDF would exceed that. Even if it fits, the LLM performs worse with very long contexts (the "lost in the middle" problem). RAG solves this by retrieving only the 4-5 most relevant small chunks.

---

## 2. What is RAG? (Retrieval-Augmented Generation)

RAG is a pattern that combines **information retrieval** with **text generation**:

```
Traditional LLM:
  Question → LLM → Answer (from training data, may hallucinate)

RAG:
  Question → Search your documents → Get relevant chunks → LLM + chunks → Answer (grounded in your data)
```

**Why RAG over fine-tuning?**

| Approach | Pros | Cons |
|----------|------|------|
| **Fine-tuning** | Model "learns" your data | Expensive, slow, needs retraining when data changes, still can hallucinate |
| **RAG** | No training needed, data can change anytime, answers are traceable to sources | Retrieval quality matters, slightly more latency |

RAG is preferred for most real-world Q&A applications because it's cheaper, more flexible, and answers are verifiable.

---

## 3. The Complete Pipeline (Deep Dive)

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────┐
│  Load    │ →  │  Split   │ →  │  Embed    │ →  │  Store in   │ →  │ Retrieve  │ →  │ Generate │
│  Docs    │    │  Chunks  │    │  Vectors  │    │  FAISS      │    │ Top-K     │    │ Answer   │
└──────────┘    └──────────┘    └───────────┘    └─────────────┘    └───────────┘    └──────────┘
```

---

### 3.1 Document Loading (`src/document_loader.py`)

**What happens:** Raw files (PDF, TXT, CSV, MD, URLs) are converted into a standard format that the rest of the pipeline can process.

**How it works:**
```python
# Every document becomes a LangChain Document object:
Document(
    page_content="The actual text content...",
    metadata={"source": "report.pdf", "page": 3}
)
```

Each file type needs a different parser:
- **PDF** (`PyPDFLoader`) — Extracts text page by page. Each page becomes a separate Document with `page` in metadata.
- **TXT** (`TextLoader`) — Reads the whole file as one Document.
- **CSV** (`CSVLoader`) — Each row becomes a separate Document.
- **Markdown** (`UnstructuredMarkdownLoader`) — Parses markdown structure.
- **URL** (`WebBaseLoader`) — Uses `BeautifulSoup` to scrape HTML and extract text.

**Interview Q: "Why use LangChain's loaders instead of just reading files with Python?"**
> LangChain loaders handle edge cases (encoding issues, PDF parsing, HTML cleaning) and produce a standard `Document` format with metadata. This metadata (source, page number) is critical later for **source citation** in answers.

---

### 3.2 Text Splitting / Chunking (`src/text_splitter.py`)

**What happens:** Large documents are broken into smaller pieces called "chunks."

**Why we need chunking:**
1. **Embedding models have token limits** — `text-embedding-3-small` works best on passages of ~500-1000 tokens.
2. **Precision** — If you embed an entire 100-page document as one vector, the embedding represents a vague "average" of the whole document. A small, focused chunk gives a much more precise embedding.
3. **LLM context** — We only want to send the most relevant chunks to the LLM, not the whole document.

**How `RecursiveCharacterTextSplitter` works:**

It tries to split text in the most "natural" way by using a hierarchy of separators:

```
Priority 1: Split on "\n\n" (paragraph breaks)
Priority 2: Split on "\n" (line breaks)
Priority 3: Split on ". " (sentences)
Priority 4: Split on " " (words)
Priority 5: Split on "" (individual characters — last resort)
```

It starts with Priority 1. If a chunk is still too large after splitting on `\n\n`, it goes to Priority 2, and so on. This preserves the most meaningful text boundaries.

**Chunk overlap explained:**

```
Chunk size = 1000, Overlap = 200

Document: [==========A==========][==========B==========][==========C==========]

Chunk 1:  [==========A==========]
Chunk 2:              [====overlap====][======B=======]
Chunk 3:                                    [====overlap====][======C=======]
```

The last 200 characters of Chunk 1 are also the first 200 characters of Chunk 2. This prevents losing context at chunk boundaries. Without overlap, a sentence split across two chunks would lose its meaning in both.

**Interview Q: "How do you choose chunk_size and chunk_overlap?"**
> It depends on the use case. Smaller chunks (300-500) give more precise retrieval but may lose context. Larger chunks (1000-2000) preserve more context but may include irrelevant info. 1000/200 is a widely-used default. In production, you'd experiment with different values and evaluate retrieval quality.

---

### 3.3 Embeddings (`src/embeddings.py`)

**What happens:** Each text chunk is converted into a **vector** (a list of floating-point numbers) that captures its semantic meaning.

**What is an embedding?**

An embedding is a numerical representation of text in high-dimensional space. Think of it as converting words into coordinates:

```
"king"  → [0.2, 0.8, 0.1, ..., 0.5]   (1536 dimensions)
"queen" → [0.21, 0.79, 0.12, ..., 0.48]  (very similar vector!)
"car"   → [0.9, 0.1, 0.7, ..., 0.2]   (very different vector)
```

Key property: **Texts with similar meaning have vectors that are close together** in this high-dimensional space.

**How `text-embedding-3-small` works:**
- It's a neural network trained by OpenAI on billions of text pairs.
- Input: Any text string (up to ~8000 tokens)
- Output: A vector of 1536 floating-point numbers
- The model learned that semantically similar texts should produce similar vectors.

**Why OpenAI embeddings vs. open-source?**
> OpenAI's models are high quality and easy to use. Open-source alternatives (like `sentence-transformers`) can run locally (no API cost, no data leaving your machine) but may have lower quality. For production, you'd consider cost, privacy, and quality trade-offs.

**Interview Q: "How do you measure similarity between two vectors?"**
> **Cosine similarity** — It measures the angle between two vectors, ignoring magnitude. Value ranges from -1 (opposite) to 1 (identical). FAISS uses L2 (Euclidean) distance by default, which for normalized vectors is equivalent to cosine similarity.
>
> ```
> cosine_similarity = dot(A, B) / (||A|| * ||B||)
> ```

---

### 3.4 Vector Store — FAISS (`src/vector_store.py`)

**What happens:** All chunk embeddings are stored in a data structure optimized for fast similarity search.

**What is FAISS?**
- Stands for **Facebook AI Similarity Search**
- Developed by Meta's AI Research team
- It's a library for efficient **nearest neighbor search** in high-dimensional vector spaces
- Runs entirely **in-memory** (very fast but limited by RAM)

**How the search works internally:**

```
Your question: "What is the company's revenue?"
     ↓
Embed the question → [0.3, 0.7, 0.2, ..., 0.4]  (1536-dim vector)
     ↓
FAISS compares this vector against ALL stored chunk vectors
     ↓
Returns the top-K closest chunks (by L2 distance or cosine similarity)
```

For small datasets (< 100K vectors), FAISS uses **brute-force search** (compares against every vector). For larger datasets, it uses **approximate nearest neighbor (ANN)** algorithms like IVF (Inverted File Index) that partition the space into clusters for faster search.

**Why FAISS over other vector databases (Pinecone, Weaviate, ChromaDB)?**

| Database | Type | Best For |
|----------|------|----------|
| **FAISS** | In-memory library | Prototyping, small-medium datasets, no server needed |
| **ChromaDB** | Embedded DB | Local persistence, easy setup |
| **Pinecone** | Cloud service | Production, large scale, managed infrastructure |
| **Weaviate** | Self-hosted/cloud | Production, hybrid search built-in |

We use FAISS because it's simple, fast, and requires no external server — perfect for a project/prototype.

**Interview Q: "What happens when the app restarts? Is the data lost?"**
> FAISS stores vectors in memory, so they're lost on restart. However, `vector_store.py` has `save_vector_store()` and `load_vector_store()` functions that serialize the index to disk. In this app's current Streamlit flow, the index is rebuilt each session. In production, you'd persist and reload it.

---

### 3.5 Retrieval (`src/retriever.py`)

**What happens:** When the user asks a question, the system finds the most relevant chunks from the vector store.

This project offers **3 retrieval strategies**:

#### a) Similarity Search (Basic)

```
Question embedding → Find K nearest chunk embeddings → Return those chunks
```

Simple and effective. But it has a weakness: if your top-4 chunks are all about the same paragraph (near-duplicates), you waste retrieval slots.

#### b) MMR — Maximal Marginal Relevance

MMR balances **relevance** and **diversity**:

```
Score = lambda * similarity(chunk, query) - (1 - lambda) * max_similarity(chunk, already_selected)
```

- `lambda = 1.0` → Pure relevance (same as similarity search)
- `lambda = 0.0` → Pure diversity (picks the most different chunks)
- This project uses `lambda = 0.7` (slightly favor relevance)

**Interview Q: "When would you use MMR over similarity?"**
> When your documents have repetitive content. For example, if a legal contract repeats the same clause in multiple sections, similarity search would return all copies of that clause. MMR would return one copy and use the remaining slots for other relevant information.

#### c) Hybrid Search (Default — the most important one)

This combines TWO different retrieval approaches:

**BM25 (Best Matching 25) — Keyword/Lexical Search:**
```
How BM25 scores a document for a query:
  - Counts how many times query terms appear in the document (Term Frequency)
  - Penalizes terms that appear in many documents (Inverse Document Frequency)
  - Normalizes by document length

Example:
  Query: "machine learning accuracy"
  Doc A: "machine learning improves accuracy significantly" → HIGH score (exact terms match)
  Doc B: "AI models perform well on benchmarks" → LOW score (no exact term matches)
```

BM25 is great at finding **exact keyword matches** but misses semantic connections (it wouldn't know that "AI models" is related to "machine learning").

**FAISS Vector Search — Semantic Search:**
```
  Query: "machine learning accuracy"
  Doc A: "machine learning improves accuracy significantly" → HIGH score
  Doc B: "AI models perform well on benchmarks" → ALSO HIGH score (semantically related!)
```

Vector search understands meaning but might miss documents that use the exact query terms in a specific technical way.

**Hybrid = BM25 + Vector Search + Ensemble:**

```python
EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # 40% keyword, 60% semantic
)
```

The `EnsembleRetriever` uses **Reciprocal Rank Fusion (RRF)** to combine results:

```
RRF Score = sum( weight_i / (k + rank_i) )

Where:
  - rank_i = position of the document in retriever i's results
  - k = constant (usually 60) to prevent high scores for top-ranked docs
  - weight_i = the weight assigned to retriever i
```

Example:
```
BM25 results:    [Doc A (rank 1), Doc C (rank 2), Doc B (rank 3)]
Vector results:  [Doc B (rank 1), Doc A (rank 2), Doc D (rank 3)]

RRF scores (k=60, weights 0.4/0.6):
  Doc A: 0.4/(60+1) + 0.6/(60+2) = 0.00656 + 0.00968 = 0.01624
  Doc B: 0.4/(60+3) + 0.6/(60+1) = 0.00635 + 0.00984 = 0.01619
  Doc C: 0.4/(60+2) + 0.0        = 0.00645
  Doc D: 0.0        + 0.6/(60+3) = 0.00952

Final ranking: Doc A > Doc B > Doc D > Doc C
```

Doc A ranks highest because it appeared in BOTH retrievers' results.

**Interview Q: "Why is hybrid better than just vector search?"**
> Vector search can miss exact keyword matches that are critical. For example, if a user asks about "Error code XJ-4012", vector search might return chunks about "error codes" in general. BM25 would find the exact chunk containing "XJ-4012". Hybrid gets the best of both worlds.

---

### 3.6 RAG Chain — The Core Logic (`src/chain.py`)

**What happens:** The user's question + retrieved chunks + chat history are assembled and sent to the LLM.

This is built using **LCEL (LangChain Expression Language)** — a way to compose chains using the `|` pipe operator (similar to Unix pipes).

#### Part 1: Question Contextualization

**Problem:** In a conversation, follow-up questions are ambiguous without history:

```
User: "What is the company's revenue?"
AI:   "The company's revenue is $5M..."
User: "How about last year?"     ← What does "last year" refer to? The retriever doesn't know!
```

**Solution:** Before retrieving, rewrite the follow-up into a standalone question:

```
Chat history + "How about last year?"
     ↓ (LLM rewrites it)
"What was the company's revenue last year?"
     ↓ (Now the retriever can find relevant chunks)
```

The code:
```python
contextualize_chain = _CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
```

This is a mini-chain: Prompt template → LLM → Extract string output.

The `contextualize_question()` function only triggers this rewrite if there IS chat history. For the first question, it passes through unchanged.

#### Part 2: The Full RAG Chain

```python
chain = (
    RunnablePassthrough.assign(
        context=lambda x: _format_docs(
            retriever.invoke(contextualize_question(x))
        )
    )
    | _RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

**Step by step:**

```
Input: {"input": "What is revenue?", "chat_history": [...]}
  │
  ▼
Step 1: RunnablePassthrough.assign(context=...)
  - Takes the input dict, keeps all existing keys
  - Adds a new "context" key by:
    a) Calling contextualize_question() → "What is revenue?"
    b) Calling retriever.invoke() → [Doc1, Doc2, Doc3, Doc4]
    c) Calling _format_docs() → Formatted string with [Source 1], [Source 2], etc.
  - Result: {"input": "What is revenue?", "chat_history": [...], "context": "[Source 1: report.pdf]..."}
  │
  ▼
Step 2: _RAG_PROMPT
  - Fills in the prompt template with {context}, {chat_history}, {input}
  - Result: A ChatPromptValue with system + history + human messages
  │
  ▼
Step 3: llm
  - Sends the messages to the LLM (OpenAI/Anthropic)
  - Result: An AIMessage with the answer
  │
  ▼
Step 4: StrOutputParser()
  - Extracts the string content from the AIMessage
  - Result: "The company's revenue is $5M according to [Source 1]..."
```

#### The RAG System Prompt

```
You are DocMind AI, a helpful assistant that answers questions
based on the provided document context. Follow these rules:
1. Answer ONLY based on the provided context.
2. If the context doesn't contain enough information, say so clearly.
3. Cite which source(s) you used (e.g., [Source 1], [Source 2]).
4. Be concise but thorough.
```

This prompt is critical for RAG quality:
- Rule 1 prevents hallucination
- Rule 2 prevents making up answers
- Rule 3 enables traceability/trust

**Interview Q: "What is LCEL and why use it?"**
> LCEL is LangChain's way of composing chains using the `|` operator. Each component in the chain implements a `Runnable` interface with `.invoke()`, `.stream()`, `.batch()` methods. Benefits: (1) Built-in streaming support, (2) Automatic async support, (3) Easy to swap components, (4) Built-in retry/fallback logic. It replaced the older `LLMChain` API.

---

### 3.7 Conversation Memory (`src/memory.py`)

**What happens:** The system remembers previous messages so users can ask follow-up questions.

```python
class ChatHistory:
    def __init__(self, max_messages: int = 20):
        self.messages: list[BaseMessage] = []
        self.max_messages = max_messages * 2  # pairs (human + AI = 2 messages)
```

**Why `max_messages * 2`?**
Each conversation "turn" has 2 messages (user + assistant). So `max_messages=20` means 20 turns = 40 message objects.

**Why trim history?**
Every message in history is sent to the LLM on each request. More history = more tokens = more cost and slower responses. Trimming keeps only the most recent conversation.

**How it's used in the pipeline:**
```
User message → Added to ChatHistory
                    ↓
              chat_history passed to chain
                    ↓
              Used for question contextualization
              AND included in the RAG prompt
                    ↓
AI response → Added to ChatHistory
```

**Interview Q: "What are alternatives to this simple memory approach?"**
> (1) **Summary memory** — Instead of storing raw messages, periodically summarize the conversation into a shorter text. (2) **Token-based trimming** — Count actual tokens instead of message count. (3) **Vector-based memory** — Store all messages in a vector store and retrieve only relevant past messages. (4) **Persistent memory** — Store in a database (Redis, PostgreSQL) so memory survives restarts.

---

## 4. The Streamlit App (`app.py`)

**Streamlit** is a Python framework that turns scripts into web apps with zero frontend code.

Key concepts used:

### `st.session_state`
Streamlit reruns the entire script on every user interaction (button click, slider change, etc.). `session_state` persists data across reruns.

```python
if "chain" not in st.session_state:
    st.session_state.chain = None       # Persists across reruns
```

Without `session_state`, the RAG chain would be lost every time the user sends a message.

### The Processing Flow

```
1. User uploads files in sidebar
2. Clicks "Process Documents"
3. process_documents() runs:
   a. Save uploaded files to temp directory
   b. Load documents → split into chunks
   c. Create embeddings → build FAISS index
   d. Create hybrid retriever
   e. Create RAG chain
   f. Store chain in session_state
4. Chat input appears
5. User types question → chain.invoke() → answer displayed
```

---

## 5. Configuration System (`config.yaml` + `src/config.py`)

Uses a **two-layer config** pattern:

```
config.yaml (defaults) → Environment variables (overrides)
```

```python
# Environment variables take priority over config.yaml
if os.getenv("LLM_PROVIDER"):
    config["llm"]["provider"] = os.getenv("LLM_PROVIDER")
```

**Why this pattern?**
- `config.yaml` has sensible defaults for development
- Environment variables override for deployment (different API keys, models per environment)
- `.env` file loads env vars locally via `python-dotenv`

---

## 6. Key Interview Questions & Answers

### Q: "What would you change for production?"

1. **Persistent vector store** — Use Pinecone/Weaviate instead of in-memory FAISS
2. **Async processing** — Document ingestion should be a background job
3. **Authentication** — Add user login and per-user document isolation
4. **Caching** — Cache embeddings and frequent queries (LangChain has built-in caching)
5. **Evaluation** — Add retrieval quality metrics (precision@k, recall, NDCG)
6. **Streaming** — Use `chain.stream()` for token-by-token output instead of waiting for the full response
7. **Error handling** — Retry logic for API failures, rate limiting
8. **Observability** — LangSmith tracing for debugging chain execution

---

## 7. Advanced Retrieval Methods (Deep Dive)

This project now supports **7 retrieval strategies**. The first 3 (similarity, MMR, hybrid) are explained in Section 3.5 above. Here are the 4 new methods:

### 7.1 Re-Ranking with Cross-Encoders (`rerank`)

**The problem:** Bi-encoder retrieval (what FAISS does) embeds the query and documents independently. This is fast but can miss nuanced relevance.

**How it works:**
```
Step 1: Retrieve top-K*3 candidates using FAISS (over-fetch)
Step 2: Score each candidate with a cross-encoder
Step 3: Re-sort by cross-encoder scores, return top-K
```

**Bi-encoder vs Cross-encoder:**

| | Bi-encoder (FAISS) | Cross-encoder (Re-ranker) |
|---|---|---|
| **Input** | Query and doc encoded separately | Query and doc encoded together |
| **Speed** | Very fast (pre-computed embeddings) | Slow (must run for each candidate) |
| **Accuracy** | Good | Better (sees query-doc interaction) |
| **Use case** | First-pass retrieval | Second-pass re-ranking |

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) processes the query and each document together through a transformer, allowing it to model fine-grained interactions between query terms and document terms. This is why we over-fetch 3x candidates — the cross-encoder can then select the truly most relevant ones.

**Interview Q: "Why not just use a cross-encoder for everything?"**
> Cross-encoders are too slow for first-pass retrieval. For 100K documents, a bi-encoder compares one query vector against 100K stored vectors (milliseconds). A cross-encoder would need to run 100K forward passes through a transformer (minutes). The two-stage approach (fast retrieval + accurate re-ranking) gives you the best of both worlds.

### 7.2 Multi-Query Retrieval (`multi_query`)

**The problem:** A single user query might not capture all the ways relevant information is phrased in the documents.

**How it works:**
```
User query: "What are the benefits of transformers?"
     ↓ (LLM generates variations)
Query 1: "What advantages do transformer models have?"
Query 2: "Why are transformers better than previous architectures?"
Query 3: "What makes the transformer architecture successful?"
     ↓ (Each query retrieves independently)
Results merged and deduplicated
```

The LLM generates 3-5 query variations that express the same information need from different angles. Each variation may surface different relevant chunks that the original query would have missed.

**Interview Q: "Doesn't this cost more LLM calls?"**
> Yes, there's one extra LLM call to generate the variations plus N retrieval calls. But the improved recall often justifies the cost. You're trading latency and cost for retrieval quality. In practice, query generation takes ~0.5s and the retrieval calls are fast, so the total overhead is small.

### 7.3 Contextual Compression (`contextual_compression`)

**The problem:** Retrieved chunks often contain both relevant and irrelevant information. Sending noisy context to the LLM wastes tokens and can confuse the model.

**How it works:**
```
Retrieved chunk: "The company was founded in 2010. Its headquarters are in
San Francisco. Revenue grew 40% in Q3 2024. The CEO enjoys hiking."
     ↓ (LLM extracts relevant parts for the query "What is the revenue?")
Compressed: "Revenue grew 40% in Q3 2024."
```

An LLM (the "compressor") reads each chunk and the query, then extracts only the sentences or phrases that are relevant to the query. This reduces noise and allows the generation model to focus on the most pertinent information.

**Interview Q: "Isn't running an LLM on every chunk expensive?"**
> It can be. The trade-off is between context quality and cost/latency. For evaluation or batch processing, it's acceptable. In production, you might use a smaller, faster model as the compressor, or only compress when chunks exceed a certain relevance threshold.

### 7.4 Parent Document Retrieval (`parent_document`)

**The problem:** Small chunks give precise retrieval but may lack surrounding context. Large chunks preserve context but reduce retrieval precision.

**How it works:**
```
Original document → Split into LARGE parent chunks (2000 chars)
Parent chunks → Split into SMALL child chunks (400 chars)
Child chunks → Embedded and indexed in FAISS

At query time:
  Query → Retrieve matching child chunks → Look up their parent chunks → Return parents
```

This is a "best of both worlds" approach:
- **Small chunks for retrieval**: The 400-char child chunks produce precise embeddings that match well against queries.
- **Large chunks for context**: The 2000-char parent chunks provide the LLM with enough surrounding context to generate complete answers.

Each child chunk stores a `parent_id` in its metadata that links back to the parent chunk it was extracted from. After retrieval, the system deduplicates parents (multiple child chunks from the same parent only return that parent once).

**Interview Q: "How is this different from just using larger chunks?"**
> Larger chunks have less precise embeddings because the embedding is an "average" of more diverse content. The parent document approach gets the precision of small-chunk embeddings for retrieval, but then swaps in the larger parent for generation. It's like using a fine-toothed comb to find what you need, then reading the full paragraph for context.

---

## 8. RAG Evaluation with RAGAS

### Why Evaluate?

Having multiple retrieval methods is useless unless you can measure which one works best for your data. The RAGAS framework (Retrieval Augmented Generation Assessment) provides automated evaluation using LLM-as-a-judge.

### The 4 RAGAS Metrics

#### Faithfulness
> "Is the answer factually consistent with the retrieved context?"

Measures whether the generated answer contains only information that can be inferred from the provided context. An answer that adds information not in the context scores low.

```
Context: "Python was created by Guido van Rossum in 1991"
Answer: "Python was created by Guido van Rossum in 1991 and is the most popular language"
Faithfulness: LOW — "most popular language" is not in the context (hallucination)
```

#### Answer Relevancy
> "Does the answer actually address the question?"

Measures how well the answer addresses the user's question. Penalizes answers that are off-topic or overly verbose with irrelevant information.

```
Question: "What year was Python created?"
Answer: "Python was created in 1991."
Relevancy: HIGH — directly answers the question

Answer: "Python is a programming language with dynamic typing and garbage collection."
Relevancy: LOW — doesn't answer when it was created
```

#### Context Precision
> "Are the relevant retrieved chunks ranked higher than irrelevant ones?"

Measures the signal-to-noise ratio of retrieval. If the retriever returns 4 chunks but only 1 is relevant, and it's ranked last, context precision is low.

#### Context Recall
> "Did the retriever find all the information needed to answer the question?"

Measures whether the retrieved contexts contain all the facts present in the ground truth answer. If the ground truth mentions 3 key facts but the retrieved contexts only cover 2 of them, recall is low.

### How RAGAS Works Internally

RAGAS uses an **LLM-as-a-judge** approach. For each metric, it prompts a language model with carefully designed evaluation prompts:

1. **Faithfulness**: The LLM breaks the answer into individual claims, then checks each claim against the context.
2. **Answer Relevancy**: The LLM generates hypothetical questions that the answer could address, then measures how similar they are to the original question.
3. **Context Precision/Recall**: The LLM compares retrieved contexts against the ground truth to determine information coverage.

### Synthetic Test-Set Generation

To evaluate without manually writing hundreds of Q&A pairs, we generate synthetic test sets:

```
Documents → LLM generates diverse Q&A pairs → Testset JSON
```

The generation process creates questions of varying difficulty: simple factual questions, questions requiring information synthesis across chunks, and questions that test whether the system correctly says "I don't know" when information is missing.

### W&B Integration

Weights & Biases (wandb.ai) provides visual dashboards to compare retrieval methods:
- **Summary metrics**: Side-by-side bar charts of all 4 RAGAS metrics across methods
- **Per-question tables**: Drill down into individual questions to see where each method fails
- **Run groups**: Each evaluation session is grouped by timestamp for historical comparison

**Interview Q: "How would you decide which retrieval method to use in production?"**
> Run the evaluation pipeline on a representative sample of your actual documents and real user queries (or a close synthetic proxy). Look at all 4 metrics: if faithfulness is low, your context is misleading the LLM. If context recall is low, you're not finding the right chunks. The "best" method depends on your data — hybrid often wins on general workloads, but re-ranking may be better for precision-critical applications.

---

### Q: "How would you evaluate RAG quality?"

1. **Retrieval metrics** — Are the correct chunks being retrieved? (Precision@K, Recall@K)
2. **Generation metrics** — Is the answer faithful to the context? (Faithfulness, Answer Relevancy)
3. **End-to-end** — Given a question, is the final answer correct? (Exact Match, F1)
4. **Tools** — RAGAS framework, LangSmith evaluation, custom test sets

### Q: "What are the limitations of this approach?"

1. **Chunk boundary problem** — Important info split across chunks may be missed
2. **Embedding quality** — If the embedding model doesn't understand domain-specific jargon, retrieval suffers
3. **No multi-modal support** — Can't process images, tables in PDFs, charts
4. **Single-turn retrieval** — Each question retrieves independently; no iterative refinement
5. **No re-ranking** — Retrieved chunks aren't re-scored for relevance (could add a cross-encoder re-ranker)

### Q: "Explain the difference between embeddings and LLMs"

| | Embeddings | LLMs |
|---|---|---|
| **Input** | Text | Text (prompt) |
| **Output** | Fixed-size vector (e.g., 1536 floats) | Variable-length text |
| **Purpose** | Represent meaning as numbers for search/comparison | Generate/understand language |
| **Model size** | Smaller (~100M params) | Much larger (~100B+ params) |
| **Cost** | Very cheap | More expensive |
| **Example** | text-embedding-3-small | GPT-4o, Claude |

### Q: "Why LangChain? Could you build this without it?"

Yes, you absolutely could. LangChain provides:
- Pre-built document loaders (saves writing PDF/CSV parsers)
- Standard interfaces (`Retriever`, `Embeddings`, `ChatModel`) so you can swap providers
- LCEL for chain composition
- Integration with 100+ tools/databases

Without LangChain, you'd directly call OpenAI's API, write your own retrieval logic, and manage prompts manually. LangChain speeds up prototyping but adds a dependency layer.
