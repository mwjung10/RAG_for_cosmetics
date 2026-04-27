# Cosmetic Ingredients QA System (Advanced RAG)

A specialized Question-Answering system focused on cosmetic ingredients (INCI), built as a university project for the "Large Language Models" course at Warsaw University of Technology.

## Project Overview
The system utilizes a **Retrieval-Augmented Generation (RAG)** architecture to provide precise, fact-based information about cosmetic substances. By connecting a quantized LLM to a dedicated vector database, the system eliminates hallucinations and provides specific chemical data, safety regulations, and functional properties.

## Architecture (Variant D)
The project implements four different versions, with **Variant D** being the most advanced:
1. **User Query** is converted into a vector using the `Harrier-oss-v1-0.6b` embedding model.
2. **Dense Retrieval**: The system fetches the Top-5 most relevant documents from a **FAISS** vector index.
3. **Re-ranking**: A `BGE-reranker-v2-m3` cross-encoder evaluates the snippets to select the Top-3 most precise contexts.
4. **Generation**: `Llama-3.1-8B-Instruct` (quantized to 4-bit NF4) generates the final answer based strictly on the provided context.

## Tech Stack
* **LLM:** Llama-3.1-8B-Instruct (via `transformers` & `bitsandbytes`)
* **Embeddings:** Harrier-oss-v1-0.6b (`sentence-transformers`)
* **Reranker:** BGE-reranker-v2-m3
* **Vector Database:** FAISS
* **Evaluation:** Hugging Face `evaluate` (ROUGE-L, chrF++)
* **Environment:** Python, PyTorch, Google Colab

## Dataset & Evaluation
* **Dataset:** Over 1,000 unique records of cosmetic ingredients (INCI).
* **Evaluation Setup:** A synthetic test set of 40 Q&A pairs generated using a "Teacher" LLM model.

## Key Features
* **4-bit Quantization (NF4):** Optimized for running large models on consumer-grade/free-tier GPUs (e.g., T4 on Colab).
* **No Chunking Strategy:** Leveraging the 128k context window of Llama-3.1 and 32k window of Harrier to maintain full document semantics.
* **Re-ranking Pipeline:** Implementation of a cross-encoder to improve retrieval precision.

## How to run
1. Clone the repository.
2. Install dependencies: `pip install transformers bitsandbytes faiss-gpu sentence-transformers evaluate`.
3. Open the provided Jupyter Notebook/Colab file.
4. Follow the steps for Data Preparation and Inference.