# Finance-AI-Chatbot-with-RAG
Built an AI chatbot using Flask, Groq API, and RAG with FAISS to provide accurate financial insights by combining real-time API data and semantic  search for contextual relevance.  Integrated real-time stock data and financial news via YFinance API and designed a responsive UI using Tailwind CSS for an enhanced user  experience.

Financial Analysis Chatbot: Technical Documentation
1. Project Overview
This project is a sophisticated financial analysis chatbot that combines:

Real-time financial data retrieval
Natural Language Processing (NLP)
Retrieval Augmented Generation (RAG)
Vector databases for knowledge storage
LLM integration through Groq
RESTful API endpoints with Flask

Core Technologies

Backend Framework: Flask with CORS and Caching
LLM Integration: Groq
Vector Database: FAISS (Facebook AI Similarity Search)
Embeddings: HuggingFace
Financial Data: yfinance API
Tool Management: LangChain

2. Architecture Components
2.1 Tool Registry System
The tool registry (tools_registry.py) implements a collection of financial analysis tools:

Company Information Tool

Retrieves comprehensive company data
Calculates key financial ratios (debt-to-equity, FCF yield)
Handles multiple exchange formats (NSE, BSE)


Dividend & Earnings Tool

Fetches calendar data for dividends
Tracks earnings release dates


Institutional Analysis Tools

Mutual fund holdings tracker
Institutional investor analysis


Technical Analysis Tool

Moving averages (SMA, EMA)
Volatility metrics
Momentum indicators (RSI, MACD)
Volume analysis



2.2 RAG System Implementation
The RAG system combines:

Vector Store: FAISS database storing financial knowledge embeddings
Query Enhancement: Contextual augmentation of user queries
Knowledge Base: Pre-loaded financial concepts and definitions

How RAG Works in This Project:
plaintextCopyUser Query → Query Enhancement → Vector Search → Context Retrieval → LLM Response
Example Process:

User asks: "What's RELIANCE.NS's P/E ratio?"
RAG system:

Retrieves relevant context about P/E ratios from vector store
Augments query with this context
Passes enhanced query to LLM


LLM uses both context and real-time data to provide informed response

2.3 Agent System
The agent system uses LangChain's agent framework:

Tool Selection

Analyzes user query
Selects appropriate financial tools
Handles multi-step queries


Response Generation

Combines tool outputs
Formats responses
Handles errors gracefully



3. Key Implementation Details
3.1 Vector Database Management
pythonCopyclass RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.initialize_vector_store()
The vector store:

Uses HuggingFace embeddings for text vectorization
Persists data locally in FAISS format
Automatically initializes with financial knowledge
Supports similarity search for context retrieval

3.2 Caching System
The project implements multi-level caching:

Flask-Caching

Caches API responses
Configurable duration
Reduces API calls to financial data sources


Tool-Level Caching

Decorated with @cache.memoize
Caches financial data requests
Optimizes repeated queries



3.3 Error Handling
Comprehensive error handling at multiple levels:

Tool Level

Handles API failures
Provides informative error messages
Implements retry logic


Agent Level

Manages tool execution failures
Provides fallback responses
Maintains conversation continuity


API Level

HTTP error handling
Request validation
Response formatting



3.4 API Endpoints
The Flask application exposes:
pythonCopy@app.route('/prompt', methods=['POST'])
def handle_prompt():
    # Handles user queries
    # Returns structured responses
Request format:
jsonCopy{
    "prompt": "What is RELIANCE.NS's current P/E ratio?"
}
Response format:
jsonCopy{
    "response": "Based on the latest data...",
    "status": "success"
}
4. Technical Deep Dives
4.1 RAG Implementation Details
The RAG system works through:

Initialization

Creates embedding model
Loads/creates vector store
Initializes with financial knowledge


Query Processing
pythonCopydef enhance_query(self, query: str) -> str:
    relevant_docs = self.vector_store.similarity_search(query, k=2)
    context = " ".join([doc.page_content for doc in relevant_docs])
    return f"Context: {context}\nQuery: {query}"

Vector Store Management

Automatic initialization
Persistence handling
Similarity search optimization



4.2 Tool Execution Flow

Query Analysis

LLM determines required tools
Extracts parameters
Plans execution steps


Tool Execution

Handles API calls
Processes raw data
Formats responses


Response Integration

Combines multiple tool outputs
Formats for user consumption
Handles follow-up context



4.3 Technical Analysis Implementation
Complex financial calculations including:

Moving Averages
pythonCopy'SMA_50': hist['Close'].rolling(window=50).mean()
'EMA_12': hist['Close'].ewm(span=12).mean()

Volatility Metrics
pythonCopy'daily_volatility': hist['Close'].pct_change().std() * np.sqrt(252)

Momentum Indicators

RSI calculation
MACD computation
Volume analysis



5. Best Practices and Optimizations

Performance Optimizations

Multi-level caching
Efficient vector search
Request batching


Security Considerations

API key management
Rate limiting
Error message sanitization


Scalability Features

Modular design
Independent components
Extensible tool registry



6. Future Enhancements
Potential improvements include:

Real-time websocket updates
Advanced technical analysis
Portfolio management features
Market sentiment analysis
Enhanced error recovery
Extended knowledge base

7. Testing and Maintenance

Testing Approaches

Unit tests for tools
Integration tests for API
Vector store validation


Monitoring

Error logging
Performance metrics
Usage statistics
