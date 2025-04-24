

import gradio as gr
import logging
import os
import time
import io # Required for StringIO log capture
from agent import ResearchAgent # Ensure agent.py is in the same directory or accessible

# --- Logging Setup for Gradio ---
# Use a distinct logger name if needed, or configure the root logger
log_stream = io.StringIO() # In-memory stream to capture logs
# Configure root logger to send INFO+ logs to the stream AND console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
                    handlers=[
                        logging.StreamHandler(), # Log to console
                        logging.StreamHandler(log_stream) # Log to our stream
                    ])
# Optional: Set level specifically for the agent's logger if it uses one
# logging.getLogger('agent_logger_name').setLevel(logging.INFO)


# --- Agent Initialization ---
try:
    # Pass max_iterations from a config or fixed value
    agent_instance = ResearchAgent(max_iterations=3)
    logging.info("ResearchAgent initialized successfully.")
except Exception as e:
    # Log critical error to console even if stream handler isn't fully working yet
    print(f"CRITICAL: Failed to initialize ResearchAgent: {e}")
    logging.critical(f"Failed to initialize ResearchAgent: {e}", exc_info=True)
    agent_instance = None

# --- Gradio Interaction Function ---
def run_research(query: str):
    """
    Function called by Gradio button click. Runs the agent and returns results.
    Yields status updates and final report/errors/logs.
    """
    if not agent_instance:
         # Use yield to update Gradio components even for errors
        yield "Error: Agent could not be initialized. Please check console logs.", "", ""

    if not query or not query.strip():
        yield "Please enter a research query.", "", ""
        return # Stop execution

    # Clear the log stream for this new run
    log_stream.seek(0)
    log_stream.truncate(0)

    logging.info(f"Gradio received query: '{query}'")
    start_time = time.time()
    final_report_content = "Starting..."
    final_error_content = ""
    final_log_content = ""

    try:
        # --- Initial Status Update ---
        yield "Processing query... (Agent is running, logs will appear after completion)", "", log_stream.getvalue()

        # --- Run the Agent ---
        # This call blocks until the agent finishes
        final_state = agent_instance.run(query.strip())

        # --- Process Final State ---
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Agent run finished in {duration:.2f} seconds.")

        final_report_content = final_state.get("final_report", "No report generated.")
        errors = final_state.get("error_log", [])

        # Format errors for display
        if errors:
            final_error_content = "\n".join(f"- {e}" for e in errors)
            logging.warning(f"Agent run completed with errors.")
        else:
            final_error_content = "None"

        # Get all logs captured during the run
        final_log_content = log_stream.getvalue()

        # --- Final Update ---
        yield final_report_content, final_error_content, final_log_content

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.critical(f"An unexpected error occurred during agent execution ({duration:.2f}s): {e}", exc_info=True)
        # Update UI with critical error info
        yield f"An critical error occurred: {e}", final_error_content, log_stream.getvalue() + f"\nCRITICAL ERROR: {e}"

# --- Gradio UI Definition ---

# Updated explanation reflecting Tavily flow
explanation_markdown = """
# Web Research Agent Demo & Explanation (Tavily Version)

This application demonstrates an AI-powered Web Research Agent built using LangGraph, Google's Gemini LLM, and the **Tavily Search API**.

## What it Does
Given a research query, the agent attempts to autonomously:
1.  **Understand** the query's intent and key topics using an LLM.
2.  **Search** the web using **Tavily** for AI-optimized results, including summaries and potential direct answers.
3.  **Evaluate** the gathered information using an LLM.
4.  **Iterate** the search process with refined queries if more information is needed.
5.  **Synthesize** the findings from Tavily into a coherent report using an LLM.

## How it Works: The Agent's Flow (LangGraph + Tavily)

The agent operates as a state machine orchestrated by **LangGraph**:

1.  **`analyze_query`**: Uses Gemini to analyze the query and generate initial search terms for Tavily.
2.  **`tavily_search`**: Uses the **Tavily API** to perform the search, retrieving ranked results with content summaries and optionally a direct answer.
3.  **`evaluate_progress`**: Uses Gemini to review the Tavily results (summaries/answer) against the original query goals. It decides whether to `continue_search` (generating new queries for Tavily) or `synthesize`.
4.  **`synthesize_report`**: Uses Gemini to combine the information gathered from Tavily into a final, structured report, citing the sources provided by Tavily.

*This flow replaces previous versions that required separate web scraping and content analysis steps.*

## Core Technologies
*   **Orchestration:** LangGraph
*   **LLM:** Google Gemini 1.5 Flash
*   **Web Search & Content:** **Tavily Search API** (`tavily-python`)
*   **UI:** Gradio

## Limitations
*   **API Dependencies:** Relies on Google Gemini and Tavily APIs.
*   **Tavily Content Quality:** Research quality depends on Tavily's results and summaries.
*   **LLM Performance:** Analysis, evaluation, and synthesis quality depend on the LLM.
*   **Factuality:** Reports information found via Tavily; doesn't independently verify facts.
*   **Iteration Limit:** Stops after a fixed number of search cycles.

## How to Use
1.  Enter your research question in the box below.
2.  Click "Run Research".
3.  Wait for processing. Status updates will appear in the "Research Report" box.
4.  The final report appears in the "Research Report" box. Any errors appear in the "Errors Log" box. **A detailed log of the agent's operations during the run appears in the "Run Logs" box *after* completion.**
"""

# Use gr.Blocks for more layout control
with gr.Blocks(theme=gr.themes.Soft(), title="Web Research Agent (Tavily)") as demo:
    gr.Markdown(explanation_markdown) # Display the updated explanation

    with gr.Row():
        query_input = gr.Textbox(label="Enter your research query:", placeholder="e.g., Recent advancements in renewable energy?", scale=4)
        submit_button = gr.Button("Run Research", scale=1)

    with gr.Column():
        # Textbox for status updates and final report
        report_output = gr.Textbox(label="Status / Research Report", lines=15, interactive=False)
        # Textbox for errors encountered
        error_output = gr.Textbox(label="Errors Log", lines=3, interactive=False)
        # Textbox for the detailed run logs
        log_output = gr.Textbox(label="Run Logs (Updates after completion)", lines=10, interactive=False)

    # Connect button click to function
    # Output components match the order yielded by run_research: report, error, log
    submit_button.click(
        fn=run_research,
        inputs=[query_input],
        outputs=[report_output, error_output, log_output] # Ensure 3 outputs match yield
    )

if __name__ == "__main__":
    if agent_instance is None:
        print("CRITICAL: Research Agent failed to initialize. Gradio UI cannot start properly.")
        # Optionally, launch Gradio with a basic error message
        # with gr.Blocks() as error_demo:
        #     gr.Markdown("# Error\n\nResearch Agent failed to initialize. Please check the console logs for details.")
        # error_demo.launch()
    else:
        print("Launching Gradio UI...")
        # Set share=False for local use, share=True for temporary public link
        demo.launch(share=False)