# Masonry - AI Web Research Agent

This project implements a Web Research Agent using Python, LangGraph, Google's Gemini LLM, and the Tavily Search API. The agent autonomously searches the web using Tavily, analyzes the results, and synthesizes the findings into a comprehensive report based on a user's query.

## Features

-   **Query Analysis:** Understands user intent, keywords, and breaks down complex queries using LLM (Gemini 1.5 Flash).
-   **AI-Optimized Web Search:** Uses the **Tavily Search API** to find relevant information, retrieve content summaries, and potentially get synthesized answers directly.
-   **Iterative Refinement:** Can perform multiple rounds of Tavily searches based on LLM evaluation to deepen research if initial results are insufficient.
-   **Information Synthesis:** Combines analyzed information (primarily from Tavily results) into a coherent final report, citing sources (URLs).
-   **Autonomous Operation:** Uses LangGraph to manage the state and flow, enabling iterative research with minimal human input.
-   **Error Handling:** Includes basic error handling for LLM calls and Tavily API interactions, logging issues encountered.
-   **Configurable Iterations:** Allows setting a maximum number of research loops (search -> evaluate).

## Architecture Overview (Tavily Flow)
![Masonry - AI Web Research Agent - visual selection](https://github.com/user-attachments/assets/959bbd17-590b-4e2b-876f-5d4f8485d045)

The agent is built using **LangGraph**, a library for building stateful, multi-actor applications with LLMs. The core logic follows a cyclical graph structure optimized for Tavily:

1.  **`analyze_query`**: (Entry Point) Takes the user's query and uses the Gemini LLM to generate an analysis (intent, keywords, etc.) and initial search terms for Tavily.
2.  **`tavily_search`**: Takes a search term, uses the `search_with_tavily` tool (wrapping the Tavily Python client) to perform the search. It requests concise content summaries and optionally a direct answer from Tavily.
3.  **`evaluate_progress`**: Uses the Gemini LLM to review the gathered information (Tavily's answer and result summaries stored in notes) against the initial query analysis. Decides whether to:
    *   **`continue_search`**: If more info is needed, generate new search terms and loop back to `tavily_search`.
    *   **`synthesize`**: If enough information is gathered or max iterations are reached.
4.  **`synthesize_report`**: (End Point) Takes all accumulated notes (from Tavily) and the original query analysis, uses the Gemini LLM to generate the final comprehensive report, citing sources provided by Tavily.

The agent's state (`AgentState`) is passed between these nodes, accumulating information throughout the process. This flow eliminates the need for separate web scraping and content analysis nodes present in previous versions.

## Technology Stack

-   **Programming Language:** Python 3.9+
-   **Agent Framework:** LangGraph
-   **LLM:** Google Gemini 1.5 Flash (via `google-generativeai` SDK)
-   **Web Search & Content:** **Tavily Search API** (via `tavily-python` library)
-   **Environment Management:** `python-dotenv`
-   **UI:** Gradio
-   **Testing:** `pytest`, `unittest.mock` (Basic tests included)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/masonry-web-research-agent.git # Replace with your repo URL
    cd masonry-web-research-agent
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` includes `langgraph`, `google-generativeai`, `python-dotenv`, `tavily-python`, `gradio`, `pytest`, etc.)*

4.  **Set Up API Keys:**
    *   Obtain a Google API Key for Gemini from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Obtain a Tavily API Key from [Tavily AI](https://tavily.com/).
    *   Create a file named `.env` in the project root directory.
    *   Add your API keys to the `.env` file:
        ```dotenv
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        TAVILY_API_KEY="YOUR_TAVILY_API_KEY_HERE"
        ```

## How to Run

**Command Line:**

Execute the agent from the command line using `main.py`. Provide the research query as an argument.

```bash
python main.py "What are the latest advancements in quantum computing?"
