# main.py
import sys
import argparse
import logging
from agent import ResearchAgent # Assuming your agent logic is in agent.py
from pprint import pprint

# Configure basic logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Web Research Agent CLI")
    parser.add_argument("query", type=str, help="The research query.")
    parser.add_argument("-i", "--iterations", type=int, default=3, help="Maximum research iterations (loops of search->scrape->analyze).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        # Set logging level to DEBUG for all modules if verbose is enabled
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    # Initialize the agent
    try:
        agent = ResearchAgent(max_iterations=args.iterations)
    except Exception as e:
        logging.critical(f"Failed to initialize Research Agent: {e}")
        sys.exit(1)

    # Run the agent
    try:
        final_state = agent.run(args.query)

        # Print the final report
        print("\n" + "="*30 + " FINAL REPORT " + "="*30)
        if final_state.get("final_report"):
            print(final_state["final_report"])
        else:
            print("No final report was generated.")
            logging.warning("Final state did not contain a 'final_report'.")

        # Optionally print errors encountered
        if final_state.get("error_log"):
            print("\n" + "="*30 + " ERRORS ENCOUNTERED " + "="*30)
            for error in final_state["error_log"]:
                print(f"- {error}")

        # Optionally print the full final state for debugging
        if args.verbose:
             print("\n" + "="*30 + " FINAL STATE (DEBUG) " + "="*30)
             # Convert set to list for pprint compatibility
             if 'visited_urls' in final_state:
                  final_state['visited_urls'] = list(final_state['visited_urls'])
             pprint(final_state)


    except Exception as e:
        logging.critical(f"An error occurred during agent execution: {e}", exc_info=True)
        print(f"\nAn critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()