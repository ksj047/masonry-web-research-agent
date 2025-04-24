# agent.py
import os
import json
import logging
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver # Keep commented unless needed

# Import the specific Tavily tool function
from tools import search_with_tavily
import prompts # Keep prompts for analysis, evaluation, synthesis
from state import AgentState # Keep state definition

# --- LLM and Logging Setup ---
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=API_KEY)
# Use Gemini 1.5 Flash - check model availability and naming conventions
# Consider error handling for model creation if needed
try:
    llm = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logging.critical(f"Failed to initialize Gemini Model: {e}")
    raise  # Re-raise the exception to stop execution if LLM fails

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Helper Function for LLM Calls ---
# agent.py

# ... (imports and setup) ...

# ... (logging) ...

# --- Helper Function for LLM Calls ---
# agent.py

# ... (imports and setup) ...

# --- Helper Function for LLM Calls ---
def call_llm(prompt: str) -> Optional[str]:
    """ Helper function to call the Gemini LLM and handle potential errors. """
    try:
        logging.debug(f"Calling LLM. Prompt length: {len(prompt)}")

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=4096,
            temperature=0.7
        )
        # safety_settings = [...] # Optional

        response = llm.generate_content(
            prompt,
            generation_config=generation_config,
            # safety_settings=safety_settings
            )

        # Enhanced response checking
        if not response.candidates:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                  reason = response.prompt_feedback.block_reason
                  logging.error(f"LLM call blocked by API. Reason: {reason}")
                  return f"Error: LLM call blocked due to {reason}"
             else:
                  logging.warning("LLM returned no candidates and no blocking reason.")
                  return None # Or return an empty string "" ?

        # *** CORE FIX: Check the NAME of the finish reason ***
        finish_reason_enum = response.candidates[0].finish_reason
        if finish_reason_enum.name != 'STOP':
             # Log details if it's not STOP
             finish_reason_val = finish_reason_enum.value
             logging.warning(f"LLM response finished with non-STOP reason: {finish_reason_enum.name} (Value: {finish_reason_val})")

             # Check safety ratings if finish reason wasn't STOP
             safety_reason = "Unknown"
             # Basic check for safety ratings presence
             if hasattr(response.candidates[0], 'safety_ratings') and response.candidates[0].safety_ratings:
                 for rating in response.candidates[0].safety_ratings:
                     # Check for a 'blocked' attribute, common in newer APIs
                     if hasattr(rating, 'blocked') and rating.blocked:
                         safety_reason = rating.category.name
                         break
                     # Add other safety check logic if needed based on API specifics
             return f"Error: LLM response ended unexpectedly (Reason: {finish_reason_enum.name}, Safety Block Detected: {safety_reason})"

        # --- If reason IS 'STOP', proceed to text extraction ---
        # Access text via the 'parts' list
        if response.candidates[0].content and response.candidates[0].content.parts:
            if response.candidates[0].content.parts:
                 result = response.candidates[0].content.parts[0].text
                 logging.debug(f"LLM response received. Length: {len(result)}")
                 return result
            else:
                 logging.warning("LLM response has content object but parts list is empty.")
                 return "" # Return empty string for valid empty responses
        else:
             logging.warning("LLM returned no content parts but finished normally (STOP reason).")
             return "" # Return empty string for valid empty responses

    except AttributeError as e:
        logging.error(f"AttributeError processing LLM response: {e}. Response structure might have changed.", exc_info=True)
        return f"Error: Failed to process LLM response structure - {e}"
    except Exception as e:
        logging.error(f"Error calling LLM: {e}", exc_info=True) # Log traceback
        return f"Error: LLM API call failed - {e}"

# --- Rest of agent.py remains the same ---
# ... (copy the rest of your existing agent.py code here) ...

# --- Rest of agent.py remains the same ---
# ... (analyze_query_node, tavily_search_node, etc.) ...

# --- Helper Function for JSON Parsing ---
def clean_json_response(llm_output: str) -> Optional[dict]:
    """ Attempts to parse JSON from LLM output, handling markdown code blocks. """
    if not llm_output or llm_output.startswith("Error:"): # Don't try to parse error messages
        return None
    try:
        # Find the start and end of the JSON block, handling potential ```json fences
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = llm_output[json_start:json_end + 1]
            # Further clean potential markdown fences if they wrap the brackets
            if json_str.strip().startswith("```json"):
                 json_str = json_str.strip()[7:]
            if json_str.strip().endswith("```"):
                 json_str = json_str.strip()[:-3]

            return json.loads(json_str.strip())
        else:
            logging.error(f"Could not find valid JSON object delimiters {{}} in LLM output: {llm_output}")
            return None

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from LLM output snippet: {e}\nOutput was: {llm_output}")
        return None # Failed to parse

# --- Agent Nodes (Adapted for Tavily) ---

def analyze_query_node(state: AgentState) -> Dict[str, Any]:
    """ Analyzes the user query to plan the research. (Mostly unchanged) """
    logging.info("Node: Analyzing Query")
    query = state['original_query']
    # Use the prompt from prompts.py as before
    try:
        prompt = prompts.QUERY_ANALYZER_PROMPT.format(query=query)
    except KeyError as e:
         logging.critical(f"KeyError during prompt formatting in analyze_query_node: {e}. Check prompts.py.")
         # Cannot proceed without a valid prompt
         return {"error_log": ["Critical prompt formatting error in analyze_query_node."]}

    llm_response = call_llm(prompt)
    parsed_analysis = clean_json_response(llm_response)

    if parsed_analysis and isinstance(parsed_analysis.get('search_queries'), list):
        logging.info(f"Query Analysis successful. Initial search queries: {parsed_analysis['search_queries']}")
        # Initialize state fields needed for the Tavily flow
        initial_updates = {
            "query_analysis": parsed_analysis,
            "search_queries": parsed_analysis['search_queries'],
            "tavily_results": [], # Store results from Tavily
            "accumulated_report_notes": [], # Store formatted Tavily result content
            "error_log": [],
            "current_iteration": 0
        }
        current_state = state.copy()
        current_state.update(initial_updates)
        return current_state # Return the whole updated state dictionary
    else:
        logging.error(f"Failed to parse query analysis from LLM. Raw Response: {llm_response}")
        error_msg = f"Failed to parse LLM response for query analysis. Raw response: {llm_response}"
        error_log = state.get("error_log", []) + [error_msg]
        # Return only the fields to update, LangGraph merges them
        return {"error_log": error_log, "search_queries": []}


def tavily_search_node(state: AgentState) -> Dict[str, Any]:
    """ Performs search using Tavily API. """
    logging.info("Node: Tavily Search")
    # Use .get() with defaults for robustness
    search_queries = state.get('search_queries', [])
    tavily_results_so_far = state.get('tavily_results', [])
    accumulated_notes = state.get('accumulated_report_notes', [])
    errors_so_far = state.get('error_log', [])

    if not search_queries:
         logging.warning("No search queries available for Tavily. Skipping search node.")
         return {} # No change if no queries

    # Use the first query in the list
    query = search_queries[0]
    remaining_queries = search_queries[1:] # Prepare for update later

    # Call Tavily - use include_answer=True to get a potential synthesized answer
    tavily_response = search_with_tavily(
        query=query,
        search_depth="basic", # Start with basic, consider "advanced" later if needed
        max_results=5,
        include_answer=True # Request Tavily's synthesized answer
    )

    # Initialize updates, default to no change
    current_errors = errors_so_far
    new_results = tavily_results_so_far
    new_notes = accumulated_notes

    # Process response
    if tavily_response and "error" not in tavily_response:
        # Get results safely using .get()
        results_list = tavily_response.get('results', [])
        tavily_answer = tavily_response.get('answer') # Can be None

        # Append raw results (optional, good for debugging)
        new_results.extend(results_list)

        # Add Tavily's answer to notes if present and not empty
        if tavily_answer:
             note = f"Tavily Answer (for query: '{query}'):\n{tavily_answer}\n---\n"
             new_notes.append(note)
             logging.info("Added Tavily's synthesized answer to notes.")

        # Add summaries from individual results to notes
        if results_list:
            for result in results_list:
                 # Safely get attributes from each result dictionary
                 url = result.get('url', 'N/A')
                 title = result.get('title', 'No Title')
                 content_summary = result.get('content', 'No Summary Provided')
                 note = f"Source: {url}\nTitle: {title}\nContent Summary: {content_summary}\n---\n"
                 new_notes.append(note)
            logging.info(f"Added {len(results_list)} result summaries to notes.")
        else:
             logging.info("Tavily returned no individual results for this query.")

    else:
        # Log Tavily API errors
        error_msg = tavily_response.get("error", f"Unknown error during Tavily search for '{query}'") if isinstance(tavily_response, dict) else f"Invalid Tavily response format for '{query}'"
        logging.error(error_msg)
        current_errors.append(error_msg)

    # Return the dictionary of fields to update
    return {
        "tavily_results": new_results,
        "accumulated_report_notes": new_notes,
        "search_queries": remaining_queries, # Update the list
        "error_log": current_errors
    }


# Removed: filter_select_urls_node, scrape_websites_node, analyze_content_node

def evaluate_progress_node(state: AgentState) -> Dict[str, Any]:
    """ Evaluates progress based on Tavily results and decides next step. """
    logging.info("Node: Evaluate Progress (Tavily Flow)")
    query = state['original_query']
    analysis = state.get('query_analysis', {}) # Get safely
    # Use notes accumulated from Tavily's answers and summaries
    notes = "\n".join(state.get('accumulated_report_notes', ["No information gathered yet."]))
    current_iter = state.get('current_iteration', 0)
    max_iter = state['max_iterations']

    # Prepare analysis JSON safely
    try:
        analysis_json = json.dumps(analysis, indent=2) if analysis else "{}"
    except TypeError as e:
        logging.error(f"Could not serialize query analysis to JSON: {e}")
        analysis_json = "{}" # Fallback

    # Prompt LLM to evaluate based on Tavily results in notes
    try:
        prompt = prompts.EVALUATOR_PROMPT.format(
            query=query,
            analysis=analysis_json,
            notes=notes,
            iteration=current_iter,
            max_iterations=max_iter
        )
    except KeyError as e:
         logging.critical(f"KeyError during prompt formatting in evaluate_progress_node: {e}. Check prompts.py.")
         # Need to make a decision even if prompt fails
         error_log = state.get("error_log", []) + [f"Critical prompt formatting error in evaluate_progress_node: {e}"]
         return {"error_log": error_log, "_decision": "stop", "current_iteration": current_iter + 1}


    llm_response = call_llm(prompt)
    parsed_eval = clean_json_response(llm_response)

    decision = "stop" # Default to stopping
    next_queries = []
    if parsed_eval and isinstance(parsed_eval.get('decision'), str): # Basic validation
        decision = parsed_eval['decision'].lower()
        if decision == 'continue':
            next_queries_raw = parsed_eval.get('next_search_queries', [])
            # Ensure next_queries is a list of strings
            next_queries = [q for q in next_queries_raw if isinstance(q, str)] if isinstance(next_queries_raw, list) else []

            if not next_queries:
                logging.warning("Evaluator decided to continue but provided no valid new queries. Will stop.")
                decision = 'stop'
            else:
                logging.info(f"Evaluator decided to continue. New queries for Tavily: {next_queries}")
        elif decision == 'synthesize':
            logging.info("Evaluator decided to synthesize.")
        else: # stop or invalid decision string
            if decision != 'stop':
                logging.warning(f"Invalid decision '{decision}' received from evaluator LLM. Defaulting to stop.")
            decision = 'stop'
            logging.info(f"Evaluator decided to stop. Reason: {parsed_eval.get('assessment', 'N/A')}")
    else:
        logging.error(f"Failed to parse evaluation response or get valid decision from LLM. Stopping. Raw: {llm_response}")
        error_log = state.get("error_log", []) + [f"Failed to parse evaluation response or get valid decision. Raw: {llm_response}"]
        return {"error_log": error_log, "_decision": "stop", "current_iteration": current_iter + 1}

    # Update state for next loop or final step
    updates = {"current_iteration": current_iter + 1, "_decision": decision}
    if decision == 'continue':
        # Prepend new queries to the list to be processed next
        current_queries = state.get('search_queries', [])
        updates["search_queries"] = next_queries + current_queries # Combine lists

    return updates


def synthesize_report_node(state: AgentState) -> Dict[str, Any]:
    """ Generates the final research report based on Tavily results. """
    logging.info("Node: Synthesize Final Report (Tavily Flow)")
    query = state['original_query']
    analysis = state.get('query_analysis', {}) # Get safely
    # Notes now contain Tavily answers/summaries
    notes = "\n".join(state.get('accumulated_report_notes', ["No information gathered."]))
    errors = "\n".join(state.get('error_log', ["None"])) # Get safely

    # Prepare analysis JSON safely
    try:
        analysis_json = json.dumps(analysis, indent=2) if analysis else "{}"
    except TypeError as e:
        logging.error(f"Could not serialize query analysis to JSON for synthesis: {e}")
        analysis_json = "{}" # Fallback

    # Use the existing SYNTHESIS_PROMPT - it takes notes and should work
    try:
        prompt = prompts.SYNTHESIS_PROMPT.format(
            query=query, analysis=analysis_json, notes=notes, errors=errors
        )
    except KeyError as e:
         logging.critical(f"KeyError during prompt formatting in synthesize_report_node: {e}. Check prompts.py.")
         fallback = f"Critical prompt formatting error during synthesis. Review notes:\n{notes}\nErrors:\n{errors}"
         error_log = state.get("error_log", []) + [f"Critical prompt formatting error in synthesize_report_node: {e}"]
         return {"final_report": fallback, "error_log": error_log}

    final_report = call_llm(prompt)

    # Check if LLM call itself returned an error string
    if final_report and final_report.startswith("Error:"):
        logging.error(f"Failed to generate final report. LLM Error: {final_report}")
        fallback = f"Failed to synthesize report due to LLM error ({final_report}). Please review accumulated notes:\n{notes}\nErrors:\n{errors}"
        error_log = state.get("error_log", []) + [f"Synthesis failed. LLM Error: {final_report}"]
        return {"final_report": fallback, "error_log": error_log}
    elif not final_report: # Handle None or empty string case
        logging.error("Failed to generate final report. LLM returned empty content.")
        fallback = f"Failed to synthesize report (LLM returned empty content). Please review accumulated notes:\n{notes}\nErrors:\n{errors}"
        error_log = state.get("error_log", []) + ["Synthesis failed: LLM returned empty content."]
        return {"final_report": fallback, "error_log": error_log}
    else:
         # Success case
        logging.info("Successfully generated final report.")
        return {"final_report": final_report}


# --- Conditional Edge Logic ---

def route_after_evaluation(state: AgentState) -> str:
    """ Determines the next node based on the evaluation decision. """
    # Use .get for safety
    decision = state.get("_decision")
    current_iter = state.get("current_iteration", 0)
    max_iter = state.get("max_iterations", 3) # Default if not set
    search_queries_left = state.get("search_queries", [])

    logging.debug(f"Routing: Decision='{decision}', Iteration={current_iter}/{max_iter}, Queries Left={len(search_queries_left)}")

    # Check if max iterations reached OR if decision is continue BUT no queries left
    if current_iter >= max_iter:
        logging.warning(f"Max iterations ({max_iter}) reached. Forcing synthesis.")
        return "synthesize" # Force synthesis
    elif decision == "continue" and not search_queries_left:
        logging.warning("Decision was 'continue' but no search queries remain. Forcing synthesis.")
        return "synthesize" # Force synthesis

    if decision == "continue":
        # Route back to Tavily search node
        return "continue_search"
    elif decision == "synthesize":
        return "synthesize"
    else: # stop or error or invalid decision string
        logging.info(f"Routing to synthesize based on decision '{decision}' or error.")
        return "synthesize"


# --- Build the Graph (Adapted for Tavily Flow) ---

def create_graph() -> StateGraph:
    """ Creates and configures the LangGraph agent with Tavily. """
    workflow = StateGraph(AgentState)

    # Add nodes for the new flow
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("tavily_search", tavily_search_node)
    # Removed: filter_select_urls, scrape_websites, analyze_content
    workflow.add_node("evaluate_progress", evaluate_progress_node)
    workflow.add_node("synthesize_report", synthesize_report_node)

    # Define edges for the new flow
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "tavily_search")
    # Removed edges related to scraping/filtering/content analysis

    # Edge from search directly to evaluation
    workflow.add_edge("tavily_search", "evaluate_progress")

    # Conditional edge from evaluation
    workflow.add_conditional_edges(
        "evaluate_progress",
        route_after_evaluation,
        {
            "continue_search": "tavily_search", # Loop back to Tavily search
            "synthesize": "synthesize_report",   # Move to synthesis
        }
    )

    workflow.add_edge("synthesize_report", END)

    # Compile the graph - consider adding checkpointing for resilience
    # memory = MemorySaver()
    # app = workflow.compile(checkpointer=memory)
    app = workflow.compile()
    logging.info("LangGraph agent graph compiled for Tavily flow.")
    return app

# --- Main Agent Class (Remains the same structure) ---
class ResearchAgent:
    def __init__(self, max_iterations=3):
        self.app = create_graph()
        self.max_iterations = max_iterations
        logging.info(f"Research Agent initialized with max_iterations={max_iterations} (Tavily Flow).")

    def run(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip(): # Check for empty/whitespace query
            logging.error("Query cannot be empty.")
            return {"error": "Query cannot be empty.", "final_report": "Error: Query cannot be empty."}

        initial_state = AgentState(
            original_query=query,
            max_iterations=self.max_iterations,
            # Initialize fields potentially used in the graph
            query_analysis=None,
            search_queries=[],
            tavily_results=[],
            accumulated_report_notes=[],
            final_report=None,
            error_log=[],
            current_iteration=0,
            # Include other keys defined in AgentState with default values
            # even if not directly used in the main Tavily flow,
            # to prevent potential key errors if accessed unexpectedly.
            search_results=[],
            urls_to_scrape=[],
            scraped_data={},
            analyzed_data={},
            visited_urls=set()
        )
        logging.info(f"Starting research for query: '{query}' (Tavily Flow)")
        # Increase recursion limit for potential loops
        config = {"recursion_limit": 50}
        final_state = {} # Initialize final_state
        try:
            final_state = self.app.invoke(initial_state, config=config)
            logging.info("Research process finished (Tavily Flow).")
        except Exception as e:
             logging.critical(f"LangGraph invocation failed: {e}", exc_info=True)
             # Populate final_state with error information
             final_state = initial_state # Start with initial state
             final_state['error_log'] = final_state.get('error_log', []) + [f"CRITICAL: Agent execution failed: {e}"]
             final_state['final_report'] = f"CRITICAL ERROR: Agent execution failed. Check logs. Error: {e}"


        # Clean up temporary keys before returning, using .pop with default None
        final_state.pop('_decision', None)

        # Ensure essential keys exist in the returned state, even if run failed early
        final_state.setdefault('final_report', "Processing failed before report generation.")
        final_state.setdefault('error_log', [])

        return final_state