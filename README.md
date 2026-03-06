# Adaptive Agentic RAG Policy Assistant

An advanced, locally hosted Retrieval-Augmented Generation (RAG) system built with LangGraph, FastAPI, and Streamlit. This project transitions from a standard linear RAG pipeline to an autonomous, multi-agent workflow capable of self-reflection, query rewriting, and parallel evaluation.

## Project Achievements: Basic RAG vs. Agentic RAG

This architecture was specifically engineered to solve the common failure points of basic RAG systems (hallucinations, redundant context, and lazy generation).

* **Benchmarked** RAG architectures, improving document retrieval accuracy from 56% to 90% by transitioning from a linear pipeline to an iterative LangGraph workflow.
* **Boosted** context precision by 20% by implementing Maximal Marginal Relevance (MMR) search, eliminating redundant document chunks and maximizing context diversity.
* **Engineered** multi-agent feedback loops to eliminate "lazy generation" on ambiguous queries, forcing the LLM to extract and output highly specific quantitative data instead of generic summaries.
* **Slashed** pipeline latency by executing parallel LLM evaluators (Information Utilization and Answer Relevance) asynchronously via `ainvoke`, cutting the evaluation bottleneck in half without compromising quality control.
* **Architected** an end-to-end local deployment using FastAPI for asynchronous agent orchestration and Streamlit for the UI, featuring real-time state tracking to visualize the agent's internal reasoning.

## The Working Process (System Architecture)

The system operates as a state machine using LangGraph, utilizing a local LLM (`gemma3:1b` via Ollama) for both generation and evaluation.

1. **Ingestion & Indexing:** * PDF documents are uploaded, chunked, and embedded using `nomic-embed-text`.
   * Vectors are stored in a local FAISS database.
2. **Retrieval Loop (Context Precision):** * A user query triggers an MMR search to fetch diverse, relevant chunks.   
   * An Evaluator LLM strictly grades the context ("PASS" or "FAIL").   
   * If the context fails, a Rewriter LLM optimizes the search query and re-queries the database.
3. **Generation & Quality Assurance Loop:** * A Drafting LLM generates an initial answer based on the verified context.   
   * Two separate Evaluator LLMs run concurrently (`ainvoke`) to check **Information Utilization** (did it miss details?) and **Answer Relevance** (did it actually answer the prompt?).   
   * If either check fails, the feedback is routed back to the Drafting LLM to force a regeneration.
4. **Delivery:** * The final, verified answer is sent to the Streamlit UI alongside the internal agent trace.

## How to Replicate and Run Locally

### Prerequisites
* Python 3.9+
* [Ollama](https://ollama.com/) installed and running locally.
* The following models pulled in Ollama:
  ```bash
  ollama run gemma3:1b
  ollama pull nomic-embed-text
1. Clone and Install Dependencies
Clone the repository and install the required Python packages.
```
git clone [https://github.com/Jayakrishna143/Agentic-RAG-Policy-Assistant.git](https://github.com/Jayakrishna143/Agentic-RAG-Policy-Assistant.git)
cd Agentic-RAG-Policy-Assistant
pip install -r requirements.txt
```
2. Configure File Paths
Ensure the directories for the vector database and temporary uploads exist, or update the paths in backend.py:
* DB_PATH = r"C:\path\to\your\project\vectordb"
* TEMP_UPLOAD_PATH = r"C:\path\to\your\project\temp_uploads"
3. Start the Backend Server (FastAPI)
Open a terminal and start the asynchronous API server.
```
fastapi run backend.py
```
The API will be available at http://127.0.0.1:8000.

4. Start the Frontend User Interface (Streamlit)
Open a second, separate terminal and launch the web interface.
```
streamlit run frontend.py
```
This will automatically open the application in your default web browser.

Author
### P Jayakrishna Reddy
