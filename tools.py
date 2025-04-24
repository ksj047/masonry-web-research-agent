# tools.py
import os
import logging
from typing import List, Dict, Any
from tavily import TavilyClient # <-- Import Tavily
from dotenv import load_dotenv

# Load environment variables to get the API key
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Tavily Search Tool ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def search_with_tavily(query: str, search_depth: str = "basic", max_results: int = 5, include_answer: bool = False, include_raw_content: bool = False) -> Dict[str, Any]:
    """
    Performs a search using the Tavily Search API.

    Args:
        query: The search query.
        search_depth: "basic" or "advanced". Advanced performs deeper research.
        max_results: The maximum number of search results to return.
        include_answer: Whether to include a synthesized answer from Tavily.
        include_raw_content: Whether to include raw scraped content for results.

    Returns:
        A dictionary containing search results, and potentially an answer
        and raw content, or an error message.
    """
    logging.info(f"Performing Tavily search for: '{query}' (depth: {search_depth}, max_results: {max_results})")
    try:
        # Use tavily_client.search method
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            # include_images=False, # Optional parameter
        )
        logging.info(f"Tavily search successful for '{query}'. Found {len(response.get('results', []))} results.")
        # response structure includes keys like 'query', 'response_time', 'answer', 'results'
        # 'results' is a list of dicts, each with 'title', 'url', 'content', 'score', 'raw_content' (if requested)
        return response

    except Exception as e:
        logging.error(f"Error during Tavily search for '{query}': {e}")
        # Return an error structure that the agent can recognize
        return {"error": f"Tavily API error: {e}", "results": []}

# --- Remove or Comment Out Old Tools ---
# def search_web(...) -> ...:
#     # ... (keep commented out or remove) ...
# def scrape_website(...) -> ...:
#     # ... (keep commented out or remove) ...