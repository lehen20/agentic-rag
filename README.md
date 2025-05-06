# 🧠 Agentic RAG System with LangGraph, LanceDB, and Ollama

This project is an **Agentic Retrieval-Augmented Generation (RAG)** pipeline built using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), [LanceDB](https://github.com/lancedb/lancedb), and [Ollama](https://ollama.com/). It uses PDF documents as knowledge base, retrieves relevant context using vector search, and answers questions using an LLM agent with tool use and grading capabilities.

---

## 🚀 Features

- 📁 PDF ingestion with metadata extraction
- 🔍 Document chunking & vector embedding via `all-MiniLM-L6-v2`
- 🧠 Retrieval using LanceDB
- 🔄 Agent decision making using LangGraph (tool vs end)
- ✅ Grading of document relevance before final generation
- 🗣️ Ollama's LLaMA3.1 used for all language tasks
- 🌐 Frontend powered by Gradio
- ⚡ Redis-backed LLM caching

---

## ⚙️ Setup Instructions

1. **Clone the repo**  
```bash
git clone https://github.com/lehen20/agentic-rag.git
cd agentic-rag


2. **Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

OR 

conda activate agent python==3.10

pip install -r requirements.txt
```
3. **Download Ollama model (ensure Ollama is running)**

```bash
ollama run llama3
```
4. **Add your PDFs**
Place all relevant PDF documents inside the /docs folder.

5. **Run the app**

```bash
python main.py