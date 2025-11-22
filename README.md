# AI Customer Support Suite

This repository contains the solution for the AI Intern Evaluation Assignment. It implements three core components of an intelligent customer support system: an Email Tagging System with customer isolation, a Sentiment Analysis Evaluator, and a Mini-RAG (Retrieval-Augmented Generation) system for Knowledge Base answering.

## 📂 Project Structure

The solution is implemented in a single Jupyter Notebook (`email-tagging-mini-system.ipynb`) divided into three distinct parts:

- **Part A:** Email Tagging Mini-System (Baseline Classifier)
- **Part B:** Sentiment Analysis Prompt Evaluation (LLM-based)
- **Part C:** Mini-RAG for Knowledge Base Answering

---

## 🚀 Part A: Email Tagging Mini-System

### Problem
Classify customer support emails into specific tags (e.g., `billing`, `bug`, `feature_request`) while strictly ensuring **customer isolation**—data or tags from Customer A must not leak to Customer B.

### Approach
I implemented a **Pattern-Based Baseline Classifier** with a strict customer-lookup constraint.
1. **Customer Isolation:** A dictionary maps `customer_id` to their allowed specific tags. During prediction, the model is mathematically restricted to choose *only* from that customer's valid tag set.
2. **Pattern Matching:** A keyword dictionary scores the subject and body.
3. **Anti-Patterns:** Common noise words (e.g., "urgent", "please", "regards") are stripped to prevent false positives based on tone rather than content.

### Error Analysis
* **Accuracy:** ~42% (Baseline).
* **Observation:** The rule-based system struggles with ambiguous phrasing (e.g., distinguishing "feature request" from "general query" without explicit keywords).
* **Future Improvement:** Replace the rule-based engine with a few-shot LLM classifier or fine-tuned BERT model that takes `[valid_tags]` as dynamic context input.

---

## 🎭 Part B: Sentiment Analysis Prompt Evaluation

### Problem
Evaluate and improve LLM prompts to consistently detect sentiment (Positive/Negative/Neutral), provide a confidence score, and generate hidden reasoning.

### Model Used
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Library:** Hugging Face `transformers` + `bitsandbytes` (4-bit quantization).

### Prompt Engineering Strategy
- **Prompt V1 (Baseline):** Generic instructions. Resulted in high confidence but occasional inability to distinguish polite complaints from neutral queries.
- **Prompt V2 (Improved):**
    - **Explicit Definitions:** Defined that "billing errors" are always *Negative* even if polite.
    - **Confidence Guardrails:** Defined specific ranges (e.g., 0.8-1.0 for clear sentiment).
    - **JSON Output:** Enforced strict JSON formatting for programmatic parsing.

**Result:** V2 provided more nuanced reasoning and correctly identified "polite frustration" as negative.

---

## 🧠 Part C: Mini-RAG (Retrieval Augmented Generation)

### Problem
Build a system that retrieves relevant Knowledge Base (KB) articles and answers user queries like "How do I configure automations?".

### Architecture
1. **Ingestion:** Loaded local markdown/text KB articles.
2. **Embeddings:** Used `sentence-transformers/all-MiniLM-L6-v2` to create vector embeddings of the articles.
3. **Retrieval:** Implemented Cosine Similarity to find the top-k relevant chunks.
4. **Generation:** Passed the retrieved context + query to **Mistral-7B** to generate a natural language answer.

### Improvement Ideas (Production)
1. **Hybrid Search:** Combine vector search (semantic) with keyword search (BM25) to catch exact technical terms.
2. **Re-ranking:** Use a Cross-Encoder (e.g., ColBERT) to re-rank the top 20 retrieved results for higher precision.
3. **Pre-computation:** Store embeddings in a vector database (ChromaDB/FAISS) rather than computing in-memory.

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.10+
- GPU (Recommended for Mistral-7B inference)

### Installation
```bash
pip install pandas numpy scikit-learn torch transformers accelerate sentencepiece bitsandbytes sentence-transformers
