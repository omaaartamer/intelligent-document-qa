from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from backend.config import settings
import os
from typing import List, Dict, Any

class VectorStore:
    """Manages document embeddings and vector similarity search using ChromaDB"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        )
    
    def has_documents(self) -> bool:
        """Check if vector store contains any documents"""
        try:
            results = self.vectorstore.get(limit=1)
            return len(results['ids']) > 0
        except:
            return False

    def clear_documents(self):
        """Remove all documents from vector store"""
        try:
            results = self.vectorstore.get()
            if results['ids']:
                self.vectorstore.delete(ids=results['ids'])
                print(f"Cleared {len(results['ids'])} documents from vector store")
            else:
                print("Vector store is already empty")
        except Exception as e:
            print(f"Error clearing vector store: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Split documents into chunks and add to vector store with embeddings"""
        all_texts = []
        all_metadatas = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({
                    "filename": doc["filename"],
                    "year": doc["year"],
                    "word_count": doc["word_count"]
                })
        
        self.vectorstore.add_texts(
            texts=all_texts,
            metadatas=all_metadatas
        )
        
        return len(all_texts)
    def search_documents(self, query: str, year_filter: int = None, k: int = 5) -> List[Dict]:
        """Perform semantic similarity search to find relevant document chunks
        
        Args:
            query: Search query text
            year_filter: Optional year to filter results by publication date
            k: Number of top results to return
            
        Returns:
            List of matching document chunks with metadata
        """
        if year_filter:
            filter_dict = {"year": year_filter}
            docs = self.vectorstore.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
        else:
            docs = self.vectorstore.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", ""),
                "year": doc.metadata.get("year", ""),
            })
        
        return results
    
    def get_available_years(self) -> List[int]:
        """Get a sorted list of unique years from document metadata"""
        try:
            results = self.vectorstore.get(include=["metadatas"])
            
            unique_years = set()
            for metadata in results.get('metadatas', []):
                if 'year' in metadata:
                    unique_years.add(metadata['year'])
            
            return sorted(list(unique_years))
        
        except Exception as e:
            print(f"Error getting available years: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get statistics about the documents in the vector store"""
        try:
            results = self.vectorstore.get(include=["metadatas"])
            metadatas = results.get('metadatas', [])
            
            if not metadatas:
                return {"doc_count": 0, "min_year": "N/A", "max_year": "N/A"}
            
            unique_files = set(m['filename'] for m in metadatas if 'filename' in m)
            unique_years = set(m['year'] for m in metadatas if 'year' in m)
            
            return {
                "doc_count": len(unique_files),
                "min_year": min(unique_years) if unique_years else "N/A",
                "max_year": max(unique_years) if unique_years else "N/A"
            }
        
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"doc_count": 0, "min_year": "N/A", "max_year": "N/A"}
