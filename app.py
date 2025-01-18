import os
from typing import TYPE_CHECKING, Dict
from typing_extensions import Annotated
from langchain_core.tools import StructuredTool,List
from datetime import date
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_caching import Cache
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
import yfinance as yf
import pandas as pd
import numpy as np
import logging

from tools_registry import tool_registry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('finance_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with caching
app = Flask(__name__)
CORS(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configuration
class Config:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    CACHE_DURATION = 3600  # Cache duration in seconds
    VECTOR_DB_PATH = "vector_store"  # This will create a directory called "vector_store"
    MODEL_NAME = "llama3-8b-8192"
    TEMPERATURE = 0.2
    MAX_RETRIES = 3
    
config = Config()

# Enhanced financial tools with caching and error handling
class FinanceTools:
    @staticmethod
    def company_information(ticker: str) -> Dict:
        """Retrieve comprehensive company information including financials and metrics."""
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.get_info()
            
            if 'totalDebt' in info and 'totalEquity' in info and info['totalEquity'] != 0:
                info['debtToEquityRatio'] = info['totalDebt'] / info['totalEquity']
            
            if 'freeCashflow' in info and 'marketCap' in info and info['marketCap'] != 0:
                info['fcfYield'] = (info['freeCashflow'] / info['marketCap']) * 100
                
            return info
        except Exception as e:
            logger.error(f"Error fetching company information for {ticker}: {str(e)}")
            raise Exception(f"Error fetching company information: {str(e)}")

    @staticmethod
    def last_dividend_and_earnings_date(ticker: str) -> Dict:
        """Retrieve company's last dividend date and earnings release dates."""
        try:
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.get_calendar()
        except Exception as e:
            logger.error(f"Error fetching dividend and earnings dates for {ticker}: {str(e)}")
            raise Exception(f"Error fetching dividend and earnings dates: {str(e)}")

    @staticmethod
    def summary_of_mutual_fund_holders(ticker: str) -> Dict:
        """Retrieve company's top mutual fund holders with share percentages."""
        try:
            ticker_obj = yf.Ticker(ticker)
            mf_holders = ticker_obj.get_mutualfund_holders()
            return mf_holders.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error fetching mutual fund holders for {ticker}: {str(e)}")
            raise Exception(f"Error fetching mutual fund holders: {str(e)}")

    @staticmethod
    def summary_of_institutional_holders(ticker: str) -> Dict:
        """Retrieve company's top institutional holders with share percentages."""
        try:
            ticker_obj = yf.Ticker(ticker)
            inst_holders = ticker_obj.get_institutional_holders()
            return inst_holders.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error fetching institutional holders for {ticker}: {str(e)}")
            raise Exception(f"Error fetching institutional holders: {str(e)}")

    @staticmethod
    def stock_grade_upgrades_downgrades(ticker: str) -> Dict:
        """Retrieve grade ratings upgrades and downgrades details."""
        try:
            ticker_obj = yf.Ticker(ticker)
            curr_year = date.today().year
            upgrades_downgrades = ticker_obj.get_upgrades_downgrades()
            upgrades_downgrades = upgrades_downgrades.loc[upgrades_downgrades.index > f"{curr_year}-01-01"]
            upgrades_downgrades = upgrades_downgrades[upgrades_downgrades["Action"].isin(["up", "down"])]
            return upgrades_downgrades.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error fetching stock grades for {ticker}: {str(e)}")
            raise Exception(f"Error fetching stock grades: {str(e)}")

    @staticmethod
    def stock_splits_history(ticker: str) -> Dict:
        """Retrieve company's historical stock splits data."""
        try:
            ticker_obj = yf.Ticker(ticker)
            hist_splits = ticker_obj.get_splits()
            return hist_splits.to_dict()
        except Exception as e:
            logger.error(f"Error fetching stock splits history for {ticker}: {str(e)}")
            raise Exception(f"Error fetching stock splits history: {str(e)}")

    @staticmethod
    def stock_news(ticker: str) -> Dict:
        """Retrieve latest news articles discussing particular stock ticker."""
        try:
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.get_news()
        except Exception as e:
            logger.error(f"Error fetching stock news for {ticker}: {str(e)}")
            raise Exception(f"Error fetching stock news: {str(e)}")

    @staticmethod
    def technical_analysis(ticker: str) -> Dict:
        """Provide technical analysis indicators."""
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="6mo")
            
            analysis = {
                'moving_averages': {
                    'SMA_50': hist['Close'].rolling(window=50).mean().iloc[-1],
                    'SMA_200': hist['Close'].rolling(window=200).mean().iloc[-1],
                    'EMA_12': hist['Close'].ewm(span=12).mean().iloc[-1],
                    'EMA_26': hist['Close'].ewm(span=26).mean().iloc[-1]
                },
                'volatility': {
                    'daily_volatility': hist['Close'].pct_change().std() * np.sqrt(252),
                    'avg_true_range': (hist['High'] - hist['Low']).mean(),
                    'bollinger_bands': {
                        'upper': hist['Close'].rolling(20).mean() + (hist['Close'].rolling(20).std() * 2),
                        'lower': hist['Close'].rolling(20).mean() - (hist['Close'].rolling(20).std() * 2)
                    }
                },
                'momentum': {
                    'rsi': calculate_rsi(hist['Close']),
                    'macd': calculate_macd(hist['Close'])
                },
                'volume_analysis': {
                    'avg_volume': hist['Volume'].mean(),
                    'volume_trend': (hist['Volume'].iloc[-5:].mean() / 
                                   hist['Volume'].iloc[-10:-5].mean() - 1) * 100,
                    'price_volume_trend': calculate_pvt(hist)
                }
            }
            return analysis
        except Exception as e:
            logger.error(f"Error performing technical analysis for {ticker}: {str(e)}")
            raise Exception(f"Error performing technical analysis: {str(e)}")

class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._initialize_tools()

    def _initialize_tools(self):
        finance_tools = FinanceTools()
        
        self._tools = {
            'company_info': StructuredTool.from_function(
                func=finance_tools.company_information,
                name="company_information",
                description="Retrieve comprehensive company information including financials and metrics."
            ),
            'dividend_earnings': StructuredTool.from_function(
                func=finance_tools.last_dividend_and_earnings_date,
                name="last_dividend_and_earnings_date",
                description="Retrieve company's last dividend date and earnings release dates."
            ),
            'mutual_fund_holders': StructuredTool.from_function(
                func=finance_tools.summary_of_mutual_fund_holders,
                name="summary_of_mutual_fund_holders",
                description="Retrieve company's top mutual fund holders with share percentages."
            ),
            'institutional_holders': StructuredTool.from_function(
                func=finance_tools.summary_of_institutional_holders,
                name="summary_of_institutional_holders",
                description="Retrieve company's top institutional holders with share percentages."
            ),
            'stock_grades': StructuredTool.from_function(
                func=finance_tools.stock_grade_upgrades_downgrades,
                name="stock_grade_upgrades_downgrades",
                description="Retrieve grade ratings upgrades and downgrades details."
            ),
            'stock_splits': StructuredTool.from_function(
                func=finance_tools.stock_splits_history,
                name="stock_splits_history",
                description="Retrieve company's historical stock splits data."
            ),
            'stock_news': StructuredTool.from_function(
                func=finance_tools.stock_news,
                name="stock_news",
                description="Retrieve latest news articles discussing particular stock ticker."
            ),
            'technical_analysis': StructuredTool.from_function(
                func=finance_tools.technical_analysis,
                name="technical_analysis",
                description="Provide technical analysis indicators."
            )
        }

    @property
    def tools(self) -> Dict[str, StructuredTool]:
        return self._tools

    @property
    def tools_list(self) -> List[StructuredTool]:
        return list(self._tools.values())

    def get_tool(self, name: str) -> StructuredTool:
        return self._tools.get(name)
tool_registry = ToolRegistry()
# Technical Analysis Helper Functions
def calculate_rsi(prices, periods=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return {
        'macd': macd.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': macd.iloc[-1] - signal_line.iloc[-1]
    }

def calculate_pvt(data):
    """Calculate Price Volume Trend."""
    pvt = ((data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * data['Volume']).cumsum()
    return {
        'current': pvt.iloc[-1],
        'trend': 'up' if pvt.iloc[-1] > pvt.iloc[-5] else 'down'
    }

# RAG Implementation
class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None
        self.initialize_vector_store()
    
    def initialize_vector_store(self):
        """Initialize or load the vector store with financial knowledge."""
        try:
            # Use Config.VECTOR_DB_PATH instead of hardcoded "faiss_index"
            vector_store_path = Path(Config.VECTOR_DB_PATH)
            
            if vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),  # Convert Path to string
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # Create the directory if it doesn't exist
                vector_store_path.mkdir(parents=True, exist_ok=True)
                
                # Initialize with comprehensive financial knowledge
                documents = [
                    "Stock markets are venues where stocks are traded.",
                    "Dividends are payments made by corporations to shareholders.",
                    "Technical analysis uses historical price and volume data to predict future movements.",
                    "Fundamental analysis evaluates a company's intrinsic value.",
                    "Market capitalization is the total value of a company's outstanding shares.",
                    "P/E ratio compares a company's stock price to its earnings per share.",
                    "Moving averages help identify trends in stock prices.",
                    "Volume indicators show the strength of price movements.",
                    "RSI measures the speed and magnitude of recent price changes.",
                    "MACD shows the relationship between two moving averages of a price."
                ]
                
                # Create new vector store
                self.vector_store = FAISS.from_texts(
                    documents,
                    self.embeddings
                )
                
                # Save it
                self.vector_store.save_local(str(vector_store_path))
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise Exception(f"Error initializing vector store: {str(e)}")
    
    def enhance_query(self, query: str) -> str:
        """Enhance user query with relevant context from vector store."""
        try:
            if not self.vector_store:
                return query
                
            relevant_docs = self.vector_store.similarity_search(query, k=2)
            context = " ".join([doc.page_content for doc in relevant_docs])
            return f"Context: {context}\nQuery: {query}"
        except Exception as e:
            logger.error(f"Error enhancing query: {str(e)}")
            return query  # Fall back to original query if enhancement fails

# Initialize components
rag_system = RAGSystem()

def initialize_agent():
    """Initialize the enhanced finance agent with error handling and RAG."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced financial advisor AI. 
        Provide detailed, accurate financial analysis using available tools.
        Always consider market context and risk factors in your responses.
        If uncertain, acknowledge limitations and suggest seeking professional advice.
        Format numbers and percentages clearly, and explain technical terms.
        When discussing stock performance, include relevant technical indicators.
        For company analysis, consider both fundamental and technical factors."""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    try:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")
            
        llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE
        )
        return create_tool_calling_agent(llm, tool_registry.tools_list, prompt)
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        raise Exception(f"Error initializing agent: {str(e)}")
# Enhanced error handling
def handle_tool_error(exception: Exception) -> str:
    """Handle tool errors with appropriate response messages."""
    error_msg = str(exception)
    logger.error(f"Tool error: {error_msg}")
    
    if "API" in error_msg:
        return "I apologize, but I'm having trouble accessing market data right now. Please try again later."
    elif "Rate limit" in error_msg:
        return "I'm experiencing high demand right now. Please try again in a few moments."
    elif "Invalid ticker" in error_msg:
        return "I couldn't find that stock symbol. Please verify the ticker and try again."
    return f"I encountered an issue while processing your request: {error_msg}"

# Initialize agent executor
try:
    agent_executor = AgentExecutor(
        agent=initialize_agent(),
        tools=tool_registry.tools_list,
        handle_tool_error=handle_tool_error,
        verbose=True
    )
except Exception as e:
    logger.error(f"Error initializing agent executor: {str(e)}")
    raise Exception(f"Error initializing agent executor: {str(e)}")

@app.route('/prompt', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')

        if not user_prompt:
            return jsonify({'error': 'Prompt is required.'}), 400

        # Enhance query using RAG
        enhanced_prompt = rag_system.enhance_query(user_prompt)

        # Process the enhanced prompt
        response = agent_executor.invoke({
            "messages": [HumanMessage(content=enhanced_prompt)]
        })

        return jsonify({
            'response': response["output"],
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"Error processing prompt: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)

