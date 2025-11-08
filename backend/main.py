from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.config import settings
from backend.document_processor import DocumentProcessor
from backend.vector_store import VectorStore
from pydantic import BaseModel
from contextlib import asynccontextmanager
from openai import OpenAI
import uvicorn
import os
import asyncio

class QuestionRequest(BaseModel):
    """Request model for question endpoint"""
    question: str
    year: int | None = None
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is political polarization?",
                "year": 2020
            }
        }

doc_processor = DocumentProcessor()
vector_store = VectorStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    print("Starting up Document Q&A system...")
    
    if vector_store.has_documents():
        print("Found existing processed documents in vector store")
        print("Ready for Q&A! (Use /reprocess endpoint to update documents)")
    else:
        print("Vector store is empty, processing documents...")
        await process_documents()
    
    yield
    
    print("Shutting down Document Q&A system...")

app = FastAPI(
    title="Intelligent Document Q&A API",
    description="""
    AI-Powered RAG System for semantic document search and question answering.
    
    This API uses Retrieval-Augmented Generation (RAG) to answer questions by:
    1. Performing semantic search across embedded PDF documents
    2. Retrieving the most relevant text chunks
    3. Using GPT to synthesize an answer from the retrieved context
    
    Features:
    - Semantic search with OpenAI embeddings
    - Year-based filtering
    - Source citations
    - Natural language Q&A
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/app")
async def serve_frontend():
    """Serve the frontend application"""
    return FileResponse('frontend/index.html')

@app.get("/")
async def root():
    """Serve the frontend application"""
    return FileResponse('frontend/index.html')

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the application is running and properly configured"""
    return {
        "status": "healthy",
        "debug_mode": settings.DEBUG,
        "api_configured": bool(settings.OPENAI_API_KEY)
    }

def _process_documents_sync():
    """Process all PDFs in docs folder and create embeddings"""
    docs_folder = "./docs"
    if not os.path.exists(docs_folder):
        print("No docs folder found")
        return
    
    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found")
        return

    print(f"Found {len(pdf_files)} PDF files")
    print("Processing documents...")
    
    documents = []
    for filename in pdf_files:
        try:
            file_path = os.path.join(docs_folder, filename)
            with open(file_path, 'rb') as file:
                content = file.read()
            
            result = doc_processor.process_document(content, filename)
            documents.append(result)
            print(f"Processed: {result['filename']} ({result['year']})")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print("Creating embeddings and storing in vector database...")
    chunk_count = vector_store.add_documents(documents)
    print(f"Ready! Processed {len(documents)} documents into {chunk_count} searchable chunks")

async def process_documents():
    """Async wrapper for document processing that runs in thread pool"""
    await asyncio.to_thread(_process_documents_sync)

@app.post("/reprocess", tags=["Admin"])
async def reprocess_documents(clear_existing: bool = True):
    """Reprocess all documents and rebuild the vector database"""
    if clear_existing:
        await asyncio.to_thread(vector_store.clear_documents)
    
    await process_documents()
    return {"message": "Documents reprocessed successfully"}
    
    
def _generate_answer_sync(question: str, year_filter: int = None):
    """Search documents and generate AI answer using retrieved context"""
    relevant_docs = vector_store.search_documents(question, year_filter=year_filter, k=5)
    
    if not relevant_docs:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the documents to answer your question.",
            "sources": [],
            "year_filter": year_filter
        }
    
    context = "\n\n".join([doc["content"] for doc in relevant_docs])
    
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Use only the information from the provided context to answer questions. If the context doesn't contain relevant information, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    sources = []
    
    no_info_phrases = [
        "couldn't find",
        "does not contain",
        "not mentioned",
        "no information",
        "unable to find",
        "context does not provide"
    ]
    
    # Check if the LLM's answer is an "I don't know" response
    answer_lower = answer.lower()
    found_info = True
    if any(phrase in answer_lower for phrase in no_info_phrases):
        found_info = False

    # Only populate sources if the LLM *found* relevant information
    if found_info:
        unique_sources = {}
        for doc in relevant_docs:
            filename = doc["filename"]
            if filename not in unique_sources:
                unique_sources[filename] = {
                    "filename": filename,
                    "year": doc["year"],
                    "preview": doc["content"][:200] + "..."
                }
        sources = list(unique_sources.values())


    return {
        "question": question,
        "answer": answer,
        "sources": sources,  # This will now be empty for irrelevant questions
        "year_filter": year_filter
    }

@app.post("/ask", tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """Answer questions using RAG: semantic search + GPT generation"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = await asyncio.to_thread(_generate_answer_sync, request.question, request.year)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/years", tags=["Metadata"])
async def get_available_years():
    """Get list of available document publication years"""
    years = await asyncio.to_thread(vector_store.get_available_years)
    return {"available_years": years}

@app.get("/stats", tags=["Metadata"])
async def get_stats():
    """Get statistics about the document collection"""
    stats = await asyncio.to_thread(vector_store.get_stats)
    return stats

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )