from langchain_tavily import TavilySearch
from langchain.tools import Tool
from dotenv import load_dotenv
import os
import ast

load_dotenv()
tavily = TavilySearch()

def simple_search(query: str) -> str:
    result = tavily.run(query)
    
    # Log raw result (optional)
    print("🔍 Tavily result:", result)

    # Try direct answer first
    if isinstance(result, dict):
        if result.get("answer"):
            return result["answer"]
        
        # Try parsing the first result's content if it's JSON-like
        results = result.get("results", [])
        if results:
            first = results[0].get("content", "")
            try:
                # Try to parse stringified dict
                data = ast.literal_eval(first)
                current = data.get("current", {})
                condition = current.get("condition", {}).get("text", "unknown")
                temp = current.get("temp_c")
                return f"The current weather is {condition} at {temp}°C."
            except Exception as e:
                pass

            # Fallback: return raw content of first result
            return first or "Could not extract weather info."

    return "No valid result found."



search_tool = Tool(
    name="search",
    func=simple_search,
    description="Use this tool to look up any current information from the internet, including weather, news, or general facts."
)


toolset = [search_tool]
