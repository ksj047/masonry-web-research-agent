# prompts.py
import json

# Basic instructions prepended to most prompts
# Define as a regular string
BASE_SYS_PROMPT = """You are an expert Web Research Agent. Your goal is to answer the user's query comprehensively by searching the web, analyzing results, and synthesizing information.
Be methodical and thorough. Cite your sources (URLs) in the final report."""

# Define as regular string, include BASE_SYS_PROMPT directly, use {placeholder}, escape literal braces with {{ }}
QUERY_ANALYZER_PROMPT = BASE_SYS_PROMPT + """
Analyze the following user query: "{query}"

**Your Task:**
1.  **Identify Intent:** What is the user trying to achieve? (e.g., find facts, get a summary, compare options, find recent news)
2.  **Extract Key Entities/Keywords:** What are the main topics, names, concepts?
3.  **Determine Information Type:** What kind of information is needed? (e.g., definitions, historical data, current events, opinions, technical specs)
4.  **Break Down (if complex):** If the query is complex, break it down into logical sub-questions.
5.  **Formulate Initial Search Queries:** Generate 1-3 effective search queries to start the research.

**Output Format:**
Provide your analysis as a JSON object with the following keys:
- "intent": string
- "keywords": list of strings
- "info_type": string
- "sub_questions": list of strings (can be empty)
- "search_queries": list of strings (1-3 queries)

**Example Query:** "What are the latest developments in AI-powered drug discovery?"
**Example Output:**
```json
{{  # Literal brace escaped
  "intent": "Find recent updates and advancements",
  "keywords": ["AI", "drug discovery", "latest developments"],
  "info_type": "Current events, technical advancements, news",
  "sub_questions": [
    "What new AI techniques are being applied to drug discovery?",
    "Which companies/research groups have reported recent breakthroughs?",
    "What are the recent challenges or successes mentioned in the news?"
  ],
  "search_queries": [
    "latest AI drug discovery news 2024",
    "AI techniques drug discovery advancements",
    "recent breakthroughs AI drug development"
  ]
}}  # Literal brace escaped
"""

CONTENT_ANALYZER_PROMPT = BASE_SYS_PROMPT +"""
Context:

Original User Query: "{query}"

Research Goal/Analysis: {analysis}

Source URL: "{url}"

Scraped Text Content:
{content}

Your Task:

Assess Relevance: Is this content relevant to the original query and research goal? (Yes/No/Partially)

Extract Key Information: Identify and list the key pieces of information directly related to the query. Be concise.

Summarize Findings: Provide a brief summary (2-4 sentences) of the relevant information found in this text.

Output Format:
Provide your analysis as a JSON object:

{{
  "relevance": "Yes/No/Partially",
  "key_info": ["Point 1", "Point 2", ...],
  "summary": "Concise summary of relevant findings..."
}}
"""

EVALUATOR_PROMPT = BASE_SYS_PROMPT +"""
Context:

Original User Query: "{query}"

Research Goal/Analysis: {analysis}

Accumulated Findings & Summaries:
{notes}

Current Iteration: {iteration} / Max Iterations: {max_iterations}

Your Task: Assess the progress towards answering the original query based on the information gathered so far.

Goal Check: Compare the accumulated findings against the query analysis (intent, keywords, sub-questions).

Are the main aspects of the query addressed?

Are the sub-questions (if any) answered?

Is the required information type present?

Sufficiency: Is the information gathered sufficient to provide a comprehensive answer? Or are there significant gaps?

Next Step Decision: Based on your assessment, decide the next course of action. Choose ONE:

"synthesize": If enough relevant information is gathered or if maximum iterations are reached.

"continue": If more information is needed and iterations remain. If choosing 'continue', suggest 1-2 new, specific search queries to fill the gaps based on your analysis.

"stop": If it seems impossible to find relevant information after trying, or if significant errors occurred (check error logs if provided).

Output Format:
Provide your decision as a JSON object:

{{
  "assessment": "Brief reasoning for your decision (1-2 sentences).",
  "decision": "synthesize" | "continue" | "stop",
  "next_search_queries": ["new query 1", "new query 2"] // Include ONLY if decision is "continue"
}}
"""

SYNTHESIS_PROMPT = BASE_SYS_PROMPT + """
Context:

Original User Query: "{query}"

Research Goal/Analysis: {analysis}

Collected & Analyzed Information (Summaries & Key Points per Source):
{notes}

Errors Encountered During Research: {errors}

Your Task:
Synthesize the collected information into a comprehensive, well-structured research report that directly answers the original user query.

Requirements:

Address the Query: Ensure the report directly answers the user's query and covers the key aspects identified in the analysis.

Combine Information: Integrate findings from multiple sources logically.

Structure: Use clear paragraphs, headings, or bullet points as appropriate. Start with a direct answer/summary if possible.

Cite Sources: For key pieces of information, mention the source URL in parentheses (e.g., "XYZ happened (source: http://example.com)"). Add a list of all used sources at the end.

Handle Contradictions: If sources conflict, note the contradiction briefly.

Acknowledge Limitations: If certain information couldn't be found or errors occurred, mention this briefly.

Tone: Objective and informative.

Generate the final research report now.
"""

