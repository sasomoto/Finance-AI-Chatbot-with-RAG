from langchain_core.tools import StructuredTool
from typing import Dict, List
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
import logging

logger = logging.getLogger(__name__)



class FinanceTools:
    @staticmethod
    def company_information(ticker: str) -> Dict:
        """Retrieve comprehensive company information including financials and metrics."""
        try:
            # Try different ticker formats if the initial one fails
            ticker_variations = [
                ticker,
                f"{ticker}.NS",  # NSE
                f"{ticker}.BO",  # BSE
                ticker.upper(),
                ticker.lower()
            ]
            
            for tick in ticker_variations:
                try:
                    ticker_obj = yf.Ticker(tick)
                    info = ticker_obj.get_info()
                    
                    # Verify we got valid data
                    if info and len(info) > 0:
                        # Add computed metrics
                        if 'totalDebt' in info and 'totalEquity' in info and info['totalEquity'] != 0:
                            info['debtToEquityRatio'] = info['totalDebt'] / info['totalEquity']
                        
                        if 'freeCashflow' in info and 'marketCap' in info and info['marketCap'] != 0:
                            info['fcfYield'] = (info['freeCashflow'] / info['marketCap']) * 100
                        
                        # Add ticker used for reference
                        info['ticker_used'] = tick
                        return info
                except:
                    continue
                    
            raise Exception(f"Could not find valid data for ticker: {ticker}. Try using exchange suffix like .NS for NSE or .BO for BSE")
                    
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
