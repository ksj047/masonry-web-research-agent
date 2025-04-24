from typing import TypedDict, List, Dict, Set, Optional

class AgentState(TypedDict):
    original_query: str
    query_analysis: Optional[dict] # Stores structured analysis from the LLM
    search_queries: List[str]      # List of search terms to use
    search_results: List[dict]     # Raw results from the search tool [{title, href, body}, ...]
    urls_to_scrape: List[str]      # URLs selected for scraping
    scraped_data: Dict[str, str]   # Mapping of URL to scraped text content
    analyzed_data: Dict[str, str]  # Mapping of URL to analysis summary
    accumulated_report_notes: List[str] # Running list of key findings/summaries
    visited_urls: Set[str]         # Keep track of processed URLs
    final_report: Optional[str]    # The final synthesized report
    error_log: List[str]           # Log errors encountered during execution
    max_iterations: int            # Safety limit for the research loop
    current_iteration: int         # Current loop count