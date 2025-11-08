# Intelligent Document Q&A System

An AI-powered RAG (Retrieval-Augmented Generation) application that provides natural language answers by performing semantic search across a large collection of PDF research documents. Built with FastAPI, OpenAI, and LangChain.

## 🎥 Demo

**[Watch the Video Demo](https://drive.google.com/file/d/1NSEKMZ6F9N-6SdQJy9ZwqwuuvJAhJfgU/view?usp=sharing)** - See the system in action!

## 🚀 Features

- **🧠 AI-Powered Answers**: Ask complex questions in natural language and receive synthesized answers from a GPT model.
- **📚 Source Citations**: Every answer is backed by citations from the original documents, preventing hallucination.
- **🔍 Advanced Semantic Search**: Uses state-of-the-art OpenAI embeddings to find the most conceptually relevant information, not just keywords.
- **📅 Dynamic Year Filtering**: Filter results by publication year to track concepts over time.
- **⚡ Modern & Robust Backend**: Built with FastAPI, using asynchronous processing to handle requests efficiently.
- **💬 Interactive Web UI**: A clean, responsive chat interface (built with vanilla HTML/CSS/JS) for easy interaction.
- **🔄 Dynamic Stats & Filters**: The frontend dynamically loads the available document count and year ranges from the backend.

## 🏗️ How the System Works (A-Z)

This project implements a complete, end-to-end Retrieval-Augmented Generation (RAG) pipeline.

### 1. Data Ingestion (Startup)

This process runs automatically the first time the server starts (or when the `/reprocess` endpoint is called).

1. **Scan**: The system scans the `/docs` folder for all PDF files.
2. **Extract**: Each PDF is processed using PyMuPDF (fitz) to extract clean, reliable text.
3. **Process Metadata**: Ideally a "smart" year extraction logic is applied. It first attempts to read the PDF's internal creation-date metadata. If that fails, it uses regex to find a 4-digit year in the filename. This, however wasn't implemented. (See Below the PDF metadata problem)
4. **Chunk**: The full text of each document is split into 1000-character chunks with a 200-character overlap (using `langchain_text_splitters`).
5. **Embed**: Each text chunk is converted into a vector (a numerical representation) using OpenAI's `text-embedding-3-large` model.
6. **Store**: All vectors and their corresponding text/metadata (filename, year) are stored in a persistent ChromaDB database on disk.

### 2. Q&A Pipeline (Per Request)

This happens every time a user clicks "Ask" in the UI.

1. **Query**: The user's question (e.g., "What is political polarization?") is sent to the `/ask` API endpoint.
2. **Retrieve**: The system embeds the user's question and queries the ChromaDB to find the `k=5` most semantically similar text chunks. If a year filter is applied, this search is narrowed to only those documents.
3. **Augment**: The text from these 5 chunks is compiled into a single "context" block.
4. **Generate**: This context is passed to the `gpt-3.5-turbo` model with a specific system prompt:
   > "You are a helpful assistant... Use only the information from the provided context to answer the question. If the context doesn't contain relevant information, say so."
5. **Respond**: The LLM's answer and the list of source documents are sent back to the frontend. The UI checks if the LLM found an answer; if not, it hides the "Sources" section to prevent showing irrelevant files.

## 💡 Architectural Decisions & Rationale

This project was built with specific, deliberate technical choices to balance performance, cost, and accuracy.

- **Embedding Model (text-embedding-3-large)**: While more expensive than other models, `text-embedding-3-large` is one of the most powerful and precise embedding models available. For academic and dense text, its ability to understand nuance is critical for retrieving the correct context, which is the most important part of the RAG pipeline.

- **LLM Choice (gpt-3.5-turbo)**: The "generation" step of RAG is less about raw knowledge and more about synthesis. Since we provide all the necessary context, `gpt-3.5-turbo` is more than capable of synthesizing a great answer. This provides a 90% reduction in token cost and a significant increase in speed compared to GPT-4, making the application practical and scalable.

- **Chunking Strategy (1000 size / 200 overlap)**: This is a robust default for RAG. A 1000-character chunk is large enough to capture one or two complete paragraphs, ensuring the vector has full semantic context. The 200-character overlap prevents a single idea from being split between two chunks, which would destroy its meaning.

- **The PDF Metadata Problem**: During development, it was clear that PDF metadata is highly unreliable. Automatically extracting `creationDate` yielded inconsistent and often inaccurate years. To ensure the year-filtering feature was 100% accurate and trustworthy, a manual curation step was required: all files have been renamed to the `YYYY_filename.pdf` format. The system's "smart" logic now uses this filename as the source of truth for the document's year, ensuring final answer accuracy.

- **Modern Library Standards**: This code has been updated to use the latest, modular LangChain libraries (e.g., `langchain_chroma`, `langchain_text_splitters`) and avoids deprecated classes. It also uses PyMuPDF (fitz) for text extraction, which is far more robust and accurate than PyPDF2.


## 🏗️ How It Works
```
PDF Documents → Text Extraction → Chunking → Vector Embeddings → ChromaDB

                                                                      ↓

User Question → Embedding → Similarity Search → GPT-3.5 → Answer + Sources

```

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Backend | FastAPI | High-performance Python web framework (async) |
| Web Server | Uvicorn | ASGI server for FastAPI |
| LLM / Embeddings | OpenAI | `gpt-3.5-turbo` (Generation), `text-embedding-3-large` (Embeddings) |
| Vector Database | ChromaDB | Open-source, persistent vector store |
| Core AI Logic | LangChain | Core RAG pipeline, chunking, and integrations |
| PDF Processing | PyMuPDF (fitz) | Fast and reliable text extraction from PDFs |
| Config | python-dotenv | Manages environment variables |
| Frontend | HTML/CSS/JS | Clean, interactive chat interface (no frameworks) |

## 📋 Prerequisites

- Python 3.10+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/omaaartamer/intelligent-document-qa.git
cd intelligent-document-qa
```

### 2. Create and Activate a Virtual Environment

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root of the project.

```bash
# Use this command...
cat > .env << EOF
OPENAI_API_KEY="your_openai_api_key_here"
EOF

# ...or create the .env file manually and add:
# OPENAI_API_KEY="your_openai_api_key_here"
```

### 5. Add Your Documents

Place all your PDF files into the `/docs` folder.

### 6. Run the Application

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open Your Browser

- **Web Interface**: http://localhost:8000
- **API Documentation (Swagger)**: http://localhost:8000/docs
- **API Documentation (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

**Note:** On the first run, the server will take a few minutes to process, embed, and store all the PDFs. Subsequent runs will be instant, as it will use the persistent database in the `./chroma_db` directory.

## 💡 Example Questions to Try

### Conceptual & Thematic Analysis

- "What is political polarization?"
- "How is 'fake news' defined in the scholarly literature?"
- "What is the 'technological performance of populism'?"
- "What is the difference between the Council of Europe's and Facebook's definition of 'Hassrede' (hate speech)?"
- "How do the papers describe the difference between 'deliberative discourse' and 'populist rhetoric'?"
- "What are the three frameworks provided in the text for why selective exposure occurs?"
- "Why is 'intense selective exposure' regarded as a threat to civil society?"

### Specific Document Retrieval (Facts & Figures)

- "What was the per-share price Elon Musk offered to purchase Twitter?"
- "According to the 2022 paper, what was Elon Musk's primary reason for wanting to buy Twitter?"
- "What legal justification did Musk's lawyers provide for terminating the $44 billion offer?"
- "What obligation does the German Netzwerkdurchsetzungsgesetz (NetzDG) place on social media platforms?"
- "What is the 'locker room talk' defense mentioned in the 2019 Pain et al. paper?"
- "What did the analysis of tweets mentioning 'Paytm' reveal?"
- "What specific problem was highlighted by the tweet mentioning '@HospitalsApollo' during demonetisation?"

### Methodology & Data Analysis

- "What 'cognitive binary' was used by Pakistani political leaders to achieve political domination?"
- "According to Pancer and Poole (2016), why did features like #hashtags and @usermentions decrease likes and retweets?"
- "How did the linguistic style of 'Leavers' and 'Remainers' differ in the 2016 Brexit referendum?"
- "What was the most frequent form of explicit persuasiveness, and in what percentage of tweets was it found?"
- "Based on the sentiment polarity for Demonetisation, what was the key inference drawn about the public's satisfaction?"

### Year-Filtered Questions

- "What did papers from 2020 say about political incivility on Twitter?"
- "Search documents from 2023. What do they say about Elon Musk's acquisition of Twitter?"
- "According to sources from 2018, what is the 'fake news' controversy?"

### RAG System & Guardrail Tests

- "When was 'Breaking Bad' released?"

*(This tests the system's ability to handle irrelevant questions. It should correctly state the information is not in the documents and provide no sources.)*

## 🔌 API Endpoints

FastAPI automatically generates interactive API documentation. Visit **http://localhost:8000/docs** to explore and test all endpoints directly in your browser.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the main index.html chat application. |
| `POST` | `/ask` | (Core API) Submits a question and returns an AI answer. |
| `GET` | `/stats` | Returns document count and year range for the UI. |
| `GET` | `/years` | Returns a sorted list of available years for the UI filter. |
| `POST` | `/reprocess` | (Admin) Clears the DB and re-processes all docs. |
| `GET` | `/health` | Health check for monitoring. |

### Example Request to `/ask`:

```bash
curl -X POST http://localhost:8000/ask \
 -H "Content-Type: application/json" \
 -d '{"question": "How does Twitter amplify politics?", "year": 2021}'
```

### Example Response:

```json
{
  "question": "How does Twitter amplify politics?",
  "answer": "According to Huszár et al. (2021), Twitter's algorithmic amplification...",
  "sources": [
    {
      "filename": "2021_Algorithmic amplification of politics on Twitter_huszár.pdf",
      "year": 2021,
      "preview": "This study investigates the algorithmic amplification of politics on Twitter..."
    }
  ],
  "year_filter": 2021
}
```

## ⚙️ Configuration

All configuration is managed via environment variables loaded from the `.env` file.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | (Required) Your OpenAI API key. | `""` |
| `EMBEDDING_MODEL` | The OpenAI model to use for embeddings. | `text-embedding-3-large` |
| `LLM_MODEL` | The OpenAI model to use for generation. | `gpt-3.5-turbo` |
| `HOST` | Server host to bind to. | `0.0.0.0` |
| `PORT` | Server port to run on. | `8000` |
| `DEBUG` | FastAPI debug mode. True enables hot-reloading. | `True` |
| `CHROMA_PERSIST_DIRECTORY` | Path to store the persistent vector database. | `./chroma_db` |

## 📄 Project Structure

```
intelligent-document-qa/
├── backend/
│   ├── main.py               # FastAPI app, endpoints, and RAG logic
│   ├── config.py             # Loads settings from .env
│   ├── document_processor.py # PDF text/metadata extraction (PyMuPDF)
│   └── vector_store.py       # ChromaDB/LangChain operations
├── frontend/
│   └── index.html            # The complete chat UI (HTML/CSS/JS)
├── docs/
│   └── (Your PDFs go here)   # e.g., 2023_report.pdf
├── chroma_db/
│   └── (Database files)      # Auto-generated by ChromaDB
├── requirements.txt          # Python dependencies
├── .env                      # Your local configuration (MUST BE CREATED)
└── README.md                 # This file
```

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

---

Built with FastAPI, OpenAI, ChromaDB, and LangChain | [Report Issues](https://github.com/omaaartamer/intelligent-document-qa/issues)
