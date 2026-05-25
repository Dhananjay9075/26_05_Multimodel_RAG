# title="app/services/search_service.py"
import requests
from typing import List, Dict, Any
from app.config import settings

class LiveWebSearchService:
    @staticmethod
    def execute_search(query: str) -> List[Dict[str, Any]]:
        """
        Interrogates internet backplanes via Tavily Engine to gather real-time data contexts.
        """
        if not settings.TAVILY_API_KEY or settings.TAVILY_API_KEY == "mock-tavily-key":
            # Graceful mock network simulation block for clean execution profiles
            return [
                {
                    "title": "Simulated Live Market Report Data",
                    "url": "https://web-context.engine/realtime-intelligence",
                    "content": f"Real-time market validation update for expression: '{query}'. Sector operational metric indexes show positive trends as of mid-2026."
                }
            ]
            
        try:
            payload = {
                "api_key": settings.TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "max_results": 3
            }
            response = requests.post("https://api.tavily.com/search", json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", "Web Intelligence Reference"),
                        "url": item.get("url", "#"),
                        "content": item.get("content", "")
                    })
                return results
            return []
        except Exception as e:
            print(f"[Search Engine Fault Notification]: {e}")
            return []