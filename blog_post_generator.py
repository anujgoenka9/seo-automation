import os
import asyncio
import csv
import re
import json
from dotenv import load_dotenv
import litellm

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as genai_types


# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    print("CRITICAL: OPENROUTER_API_KEY not found. Please set it in a .env file. Exiting.")
    exit(1) 

# CSV File Paths
BASE_FILE_PATH = "." 
CLUSTERS_CSV_PATH = os.path.join(BASE_FILE_PATH, "Clusters.csv")
POSTED_CSV_FILENAME = "Posted.csv"
POSTED_CSV_PATH = os.path.join(BASE_FILE_PATH, POSTED_CSV_FILENAME)
BLOG_OUTPUT_DIR = os.path.join(BASE_FILE_PATH, "generated_blog_posts")

# CSV Headers
CLUSTER_FIELD_CLUSTER = "Cluster"
CLUSTER_FIELD_INTENT = "Intent"
CLUSTER_FIELD_KEYWORDS = "Keywords"
CLUSTER_FIELD_PRIMARY_KEYWORD = "Primary Keyword"
CLUSTER_FIELD_COMPLETED = "Completed"

# LLM Model Names (All via OpenRouter now)
PRELIM_PLAN_MODEL_NAME = "google/gemini-2.5-flash-preview:thinking"
RESEARCH_AGENT_MODEL_NAME = "perplexity/llama-3.1-sonar-large-128k-online"
DETAILED_PLAN_MODEL_NAME = "google/gemini-2.5-flash-preview:thinking"
WRITE_BLOG_MODEL_NAME = "anthropic/claude-3.7-sonnet"
INTERNAL_LINK_MODEL_NAME = "google/gemini-2.5-flash-preview:thinking"
HTML_CONVERSION_MODEL_NAME = "openai/gpt-4o"

# --- Prompt Definitions ---
# Agent 1: Preliminary Blog Post Plan
PROMPT_AGENT_1_PRELIMINARY_PLAN = """
You are part of a team that creates world class blog posts.

For each new blog post project, you are provided with a list of keywords and search intent.

- Keywords: The keywords are to what the blog post is meant to rank for. They are scattered throughout the blog and define the topic of the blog post.
- Search intent: The search intent recognises the intent of theuser when searching up the keyword which defines be the theme of the blog post, so they click on our blog to satisfy their search.
- Primary keyword: Out of the keywords, there is one keyword known as the primary keyword. The primary keyword will go in the title and first few sentences. It is important that the topic of the blog post is related to the primary keyword so that you can place it into the title and introduction naturally.

Given a list of keywords and search intent, your job is to understand the goal of th e blog post, identify the thought process behind the flow of the blog post and come up with a preliminary plan for the post.

Your output must:
- Recognise the discussion points of the blog post.
- Be in dot point format.

You must ensure that the plan created satisfies the search intent and revolves directly around the given keywords.
When making the plan keep in mind that all keywords must be used in the final blog post.
The final goal of the project is to create a high quality, high value, highly relevant blog post that will satisfy the users search intent and give them everything they need to know about the topic.

A new project just came across your desk with below keywords and search intent:

Keywords:
{keywords}

Search intent:
{intent}

Primary keyword:
{primary_keyword}

Create the preliminary plan.
"""

# Agent 3: Detailed Plan Generation
PROMPT_AGENT_3_DETAILED_PLAN = """\
You are part of a team that creates world class blog posts.

For each new blog post project, you are provided with a list of keywords, a primary keyword, search intent, research findings and a preliminary blog post plan. Here's a definition of each of the inputs:

- Keywords: These are the keywordswhich the blog post is meant to rank for on SEO. They should be scattered throughout the blog post intelligently to help with SEO.
- Search intent: The search intent recognises the intent of the user when searching up the keyword. Our goal is to optimise the blog post to be highly relevant and valuable to the user, as such the search intent should be satisfied within the blog post.
- Research findings: This is research found from reputable sources in relation to the blog post. You must intelligently use this research to make your blog post more reputable. If specific source URLs are provided with the research (e.g., as `...text... - source: URL`), **preserve this exact format** in your plan. The copywriter will use this format directly.
- Preliminary plan: Very basic plan set out by your colleague to kick off the blog post.
- Primary keyword: Out of the keywords, there is one keyword known as the primary keyword. The primary keyword is the keyword which has the highest SEO importance and as such must go in the title and first few sentences of the blog post. It is important that the blog post is highly relevant to the primary keyword, so that it could be placed naturally into the title and introduction sections.

Given the said info, you must create a detailed plan for the blog post.

Your output must:
- Include a plan for the blog post.
- Be in dot point format.
- In each part of the blog post, you must mention which keywords should be placed.

Here are the other things you must consider:
- All keywords must be placed inside the blog post. For each section, mention which keywords to include. The keyword placement must feel natural and must make sense.
- You must include all research points in the blog post. When including research points, if they contain a source URL marker like ` - source: URL`, **ensure this marker is kept intact and in the same format** in your output plan.
- You must ensure that the plan created satisfies the search intent and revolves directly around the given keywords.
- Your plan must be very detailed.
- Keep in mind the copywriter that will use your plan to write the blog post is not an expert in the topic of the blog post. So you should give them all the detail required so they can just turn it into nicely formatted paragraphs. So your plan should include all technical detail regarding each point to be in the blog post. For example instead of saying "define X", you must have "define X as ...".
- The plan you create must have a flow that makes sense.
- You must ensure the blog post will be highly detailed and satisfy the most important concepts regarding the topic.

A new project has just came across your desk with below details:

Keywords:
{keywords}

Search intent:
{intent}

Preliminary plan:
{preliminary_plan}

Research findings:
{research_findings}

Primary keyword:
{primary_keyword}

Create the detailed plan.

Your output must only be the plan and nothing else.
"""

# Agent 4: Write Blog Post
PROMPT_AGENT_4_WRITE_BLOG = """\
You are part of a team that creates world class blog posts.
You are the team's best copywriter and are responsible for writing out the actual blog post.

For each new blog post project, you are provided with a detailed plan and research findings.
Your job is to create the blog post by closely following the detailed plan.

The blog post you create must:
- Follow the plan bit by bit.
- Use short paragraphs.
- Use bullet points and subheadings with keywords where appropriate.
- Not have any fluff. The content of the blog must be value dense and direct.
- Be very detailed.
- Include the keywords mentioned in each section within that section.
- Use the research as advised by the plan. If the plan or research findings include citation markers like ` - source: URL`, **ensure these markers are preserved exactly as they appear** in the final blog post text. Do not alter or remove them. They will be converted to links later.
- Place the primary keyword: '{primary_keyword}' in the blog title, H1 header, and early in the introduction.
- Place one keyword for each section in the heading of that section, if specified in the plan.
- When possible, pepper synonyms of the keywords throughout each section.
- When possible, use Latent Semantic Indexing (LSI) keywords and related terms to enhance context.
- Be at minimum 2000 to 2500 words long.
- Be suitable for a year 5 reading level.

Make sure to create the entire blog post draft in your first output. Don't stop or cut it short.

Here are the details of your next blog post project:

Detailed Plan:
{detailed_plan}

Research Findings:
{research_findings}

Primary Keyword:
{primary_keyword}

Write the blog post.
"""

# Agent 5: Add Internal Links
PROMPT_AGENT_5_INTERNAL_LINKS = """
You are part of a team that creates world class blog posts.
You are in charge of internal linking between blog posts.

You will be given the current DRAFT blog post and a list of PREVIOUSLY POSTED blog posts from our own website.
Each previously posted blog entry will include its URL, keywords, and a summary.

Your job is to:
1. Read the current DRAFT blog post.
2. Review the list of PREVIOUSLY POSTED blog posts.
3. Identify at least 2 highly relevant internal linking opportunities.
   - An opportunity is relevant if the topic/keywords of a PREVIOUSLY POSTED blog align well with a section or phrase in the DRAFT blog post.
   - Only make a link if it genuinely adds value for the reader and makes contextual sense. Do not force links.
4. Update the DRAFT blog post by inserting the URLs of the chosen PREVIOUSLY POSTED blogs at the most relevant locations.
   - Place the URL directly next to the keyphrase or sentence it should be associated with. For example: "...this is a key concept [Previous Blog URL here] that you can learn more about."
   - The copywriter or HTML converter will later use these URLs to create actual hyperlinks for the keyphrases. Your placement is critical.
5. Ensure you DO NOT remove any existing content or URLs (like external research source URLs already in the draft). You ONLY ADD new internal linking URLs.

Your output MUST be the complete DRAFT blog post with the newly added internal linking URLs. Do not output any other commentary, explanations, or lists of links separately.

Current DRAFT Blog Post:
{current_blog_post_content}

Previously Posted Blog Posts (for internal linking):
{internal_linking_data}

Add the internal links to the current DRAFT blog post and output the revised full text.
"""

# Agent 6: Convert to HTML
PROMPT_AGENT_6_HTML_CONVERSION = """\
You are an expert HTML coder specializing in formatting blog posts for WordPress.
You will be given the final text of a blog post, which includes placeholder URLs for both external research and internal links.
Your task is to convert this blog post into a single, well-structured HTML document.

Follow these rules precisely:
1.  **Main Container**: Wrap the entire blog post content in a single `<div>` with the inline style: `background-color: #333333; color: #ffffff; font-family: Arial, sans-serif; line-height: 1.6; padding: 20px;` (Added background-color and padding).
2.  **Global Styles for Text**: Add a `<style>` block inside the main container:
    `<style> p, .wp-block-paragraph, ul.wp-block-list, li {{ color: #ffffff !important; font-size: 20px !important; }} a {{ color: #00c2ff !important; text-decoration: underline !important; }} </style>`
3.  **Hyperlinks**:
    *   Identify all URLs in the text. These might be standalone or explicitly marked (e.g., "...keyphrase - source: http://example.com" or "...another point [http://internallink.com]").
    *   Convert these into functional `<a>` tags.
    *   If a URL is next to a keyphrase (e.g., "more about Topic X - source: http://example.com" or "read about Y [http://internallink.com]"), the keyphrase itself should become the clickable text. For example: `<a href="http://example.com">more about Topic X</a>` or `<a href="http://internallink.com">read about Y</a>`.
    *   If a URL is standalone, use the URL itself or a concise placeholder like "[source]" or "[link]" as the clickable text.
    *   All hyperlinks must be styled blue (#00c2ff) and underlined (as per the global style).
4.  **Headings (H1, H2, H3, etc.)**:
    *   Identify headings in the blog post. Assume the main blog title will be H1. Subsections should use H2, H3, etc.
    *   Apply an inline style to all headings: `border-bottom: 2px solid #00c2ff; padding-bottom: 5px; color: #ffffff;`.
5.  **Structure and Readability**:
    *   Use `<p>` tags for paragraphs.
    *   Use `<ul>` and `<li>` for bullet points.
    *   Insert `<br><br>` (double line breaks) between major sections (e.g., after a heading and its content, before the next heading) to improve visual separation.
    *   Add an "Estimated reading time: X minutes" paragraph near the beginning if such information is inferable or provided in the text. Style it like other paragraphs.
    *   If there's a "Key Takeaways" section, format it as a `<h2>` followed by a `<ul>`.
    *   If there's a "Table of Contents", format it as a `<h2>` followed by a `<ul>` of links to section IDs (you'll need to add `id` attributes to your headings for this to be functional, e.g., `<h2 id="section1-slug">Section 1 Title</h2>`).
    *   If there's an "FAQ" section, format it as a `<h2>` followed by a series of `<h3>` or `<p><strong>Question:</strong></p><p>Answer:</p>` pairs. FAQ questions (if part of links in a TOC or standalone clickable) should also be blue.
6.  **No Emojis**.
7.  **Output ONLY the HTML code.** Do not include any other text, explanations, or markdown code fences (e.g., do not wrap your code in three backticks followed by 'html').

Blog post: 

{{ $json.message.content }}

Here's an example of a well formatted output:

<div style="background-color: #333333; color: #ffffff; font-family: Arial, sans-serif; line-height: 1.6; padding: 20px;"> <style> p, .wp-block-paragraph, ul.wp-block-list, li {{ color: #ffffff !important; font-size: 20px !important; }} a {{ color: #00c2ff !important; }} </style> <h1 id="h-devin-ai-the-hype-and-reality-of-an-ai-software-engineer" class="wp-block-heading" style="border-bottom: 2px solid #00c2ff; padding-bottom: 5px;">Devin AI: The Hype and Reality of an AI Software Engineer</h1> <br><br> <p class="estimated-reading-time" style="color: #ffffff; font-size: 20px !important;">Estimated reading time: 5 minutes</p> <br><br> <h2 id="h-key-takeaways" class="wp-block-heading" style="border-bottom: 2px solid #00c2ff; padding-bottom: 5px;"><strong>Key Takeaways</strong></h2> <br><br> <ul class="wp-block-list"> <li><mark style="background-color: #ffd966;"><strong>Devin AI</strong></mark> claims to be the world's first fully autonomous AI software engineer.</li> <br><br> <li>Initial demos and claims have generated significant <mark style="background-color: #ffff00;">hype</mark> and interest.</li> <br><br> <li>Critics argue some capabilities may be exaggerated or misleading.</li> <br><br> <li>Real-world testing reveals both <em>strengths</em> and <em>limitations</em>.</li> <br><br> <li>The true impact on software engineering remains to be seen.</li> </ul> <br><br> <div class="wp-block-yoast-seo-table-of-contents yoast-table-of-contents"> <h2 style="color: #ffffff; border-bottom: 2px solid #00c2ff; padding-bottom: 5px;">Table of contents</h2> <br><br> <ul> <li><a href="#h-devin-ai-the-hype-and-reality-of-an-ai-software-engineer" data-level="1">Devin AI: The Hype and Reality of an AI Software Engineer</a></li> <br><br> <li><a href="#h-key-takeaways" data-level="2">Key Takeaways</a></li> <br><br> <li><a href="#h-what-is-devin-ai" data-level="2">What is Devin AI?</a></li> <br><br> <li><a href="#h-the-hype-around-devin-ai" data-level="2">The Hype Around Devin AI</a></li> <br><br> <li><a href="#h-putting-devin-to-the-test" data-level="2">Putting Devin to the Test</a></li> <br><br> <li><a href="#h-the-reality-check" data-level="2">The Reality Check</a></li> <br><br> <li><a href="#h-the-future-of-ai-in-software-development" data-level="2">The Future of AI in Software Development</a></li> <br><br> <li><a href="#h-frequently-asked-questions" data-level="2">Frequently Asked Questions</a></li> </ul> </div> <br><br> <p>Devin AI has burst onto the tech scene, promising to revolutionize software development as we know it. But does this AI-powered coding assistant live up to the hype? Let's dive into what Devin AI really is, what it can do, and what developers are saying after putting it to the test.</p> <br><br> </div>

Blog Post Content to Convert:
{final_blog_post_content_with_links}

Generate the HTML code.
"""

# --- Helper Functions ---

def fix_links(raw_json_response: dict) -> str:
    if not isinstance(raw_json_response, dict):
        print("Error in fix_links: raw_json_response is not a dictionary.")
        return raw_json_response.get('choices', [{}])[0].get('message', {}).get('content', "Raw response was not a dictionary.")

    try:
        text_to_process = raw_json_response['choices'][0]['message']['content']
        citations = raw_json_response.get('citations')

        if not isinstance(citations, list) or not citations:
            print("Warning in fix_links: Top-level 'citations' field is missing, not a list, or empty. Checking message.annotations...")
            annotations = raw_json_response.get('choices', [{}])[0].get('message', {}).get('annotations')
            if isinstance(annotations, list):
                extracted_urls = []
                for ann in annotations:
                    if isinstance(ann, dict) and ann.get('type') == 'url_citation':
                        url_data = ann.get('url_citation')
                        if isinstance(url_data, dict) and 'url' in url_data:
                            extracted_urls.append(url_data['url'])
                if extracted_urls:
                    print(f"Found {len(extracted_urls)} URLs in 'message.annotations'. Using these.")
                    citations = extracted_urls
                else:
                    print("Still no citation URLs found in 'message.annotations'. No links will be fixed.")
                    return text_to_process
            else:
                 print("No citation URLs found in 'message.annotations' either. No links will be fixed.")
                 return text_to_process

        modified_text = text_to_process
        for i, url in enumerate(citations):
            placeholder = f"[{i+1}]"
            replacement = f" - source: {url}"
            modified_text = re.sub(re.escape(placeholder), replacement, modified_text)

        return modified_text
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing raw_json_response in fix_links: {e}. Details: {raw_json_response}")
        return raw_json_response.get('choices', [{}])[0].get('message', {}).get('content', "Error during link fixing, original content unavailable.")

async def fetch_and_process_research_directly(query: str, client_name_for_log: str = "OnlineResearchPerplexity") -> dict:
    """
    Performs online research using a direct LiteLLM call to a Perplexity model,
    processes citations, and returns the text content and any error.
    Returns: {"text_content": str, "error": Optional[str]}
    """
    print(f"\n--- Direct LiteLLM Call for Research ({client_name_for_log}) ---")
    print(f"Research Query: '{query[:200]}...'")

    research_instruction_for_perplexity = """
You are an expert research assistant. Your task is to perform a comprehensive web search based on the user's query.
Leverage your online capabilities to:
1. Understand the core subject and information needs from the user's query.
2. Conduct a thorough search of the internet for relevant, factual, and up-to-date information.
3. Synthesize the gathered information into a coherent and detailed summary of findings.
4. Focus on extracting key points, supporting details, and any notable statistics or examples.
IMPORTANT: Your response will be processed to extract citation URLs. Ensure your output uses bracketed numerical citations (e.g., [1], [2]) and that the corresponding source URLs are available in the API response's 'citations' list or 'message.annotations'.
Your output should be the synthesized summary of your research findings.
"""
    
    messages = [
        {"role": "system", "content": research_instruction_for_perplexity},
        {"role": "user", "content": query}
    ]

    try:
        response_obj = await litellm.acompletion(
            model="openrouter/" + RESEARCH_AGENT_MODEL_NAME,
            messages=messages,
            api_key=OPENROUTER_API_KEY,
        )

        if not response_obj:
            error_msg = f"{client_name_for_log}: LiteLLM acompletion returned None."
            print(f"Error: {error_msg}")
            return {"text_content": "Research failed: No response from LLM.", "error": error_msg}

        raw_response_dict = response_obj.model_dump()
        
        fixed_content = fix_links(raw_response_dict)
        
        if "Error during link fixing" in fixed_content or "Raw response was not a dictionary" in fixed_content :
             original_content_from_raw = raw_response_dict.get('choices', [{}])[0].get('message', {}).get('content', "No text content in raw response.")
             print(f"Warning ({client_name_for_log}): fix_links reported an issue or made no changes. Using text directly from raw response if available.")
             return {"text_content": original_content_from_raw, "error": fixed_content if "Error" in fixed_content else None}

        print(f"{client_name_for_log}: Research successful and links processed.")
        return {"text_content": fixed_content, "error": None}

    except litellm.exceptions.APIConnectionError as e:
        error_msg = f"{client_name_for_log}: LiteLLM APIConnectionError: {e}"
        print(f"Error: {error_msg}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'): 
            print(f"Error details: {e.response.text}")
        return {"text_content": "Research failed due to API connection error.", "error": error_msg}
    except Exception as e:
        error_msg = f"{client_name_for_log}: An unexpected error occurred during research: {e}"
        print(f"Error: {error_msg}")
        return {"text_content": "Research failed due to an unexpected error.", "error": error_msg}

def get_next_cluster_to_process() -> dict | None:
    try:
        with open(CLUSTERS_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if CLUSTER_FIELD_COMPLETED not in reader.fieldnames:
                print(f"Error: '{CLUSTER_FIELD_COMPLETED}' column not found in {CLUSTERS_CSV_PATH}.")
                return None
            
            for row in reader:
                completed_status = row.get(CLUSTER_FIELD_COMPLETED, "").strip().lower()
                if completed_status == 'no' or not completed_status:
                    required_keys = [CLUSTER_FIELD_KEYWORDS, CLUSTER_FIELD_INTENT, CLUSTER_FIELD_PRIMARY_KEYWORD]
                    if not all(key in row for key in required_keys):
                        print(f"Warning: Skipping row due to missing one of {required_keys}. Row: {row}")
                        continue
                    print(f"Found cluster to process: {row.get(CLUSTER_FIELD_CLUSTER, 'N/A')}")
                    return row
        print(f"No clusters found with 'Completed' as 'No' or empty in {CLUSTERS_CSV_PATH}.")
        return None
    except FileNotFoundError:
        print(f"Error: {CLUSTERS_CSV_PATH} not found.")
        return None
    except Exception as e:
        print(f"Error reading {CLUSTERS_CSV_PATH}: {e}")
        return None

def update_cluster_status(cluster_primary_keyword: str, new_status: str = "Yes") -> bool:
    """
    Updates the 'Completed' status of a specific cluster in Clusters.csv.
    Identifies the cluster row by its 'Primary Keyword'.
    """
    print(f"Attempting to update status for cluster with Primary Keyword '{cluster_primary_keyword}' to '{new_status}'.")
    if not os.path.exists(CLUSTERS_CSV_PATH):
        print(f"Error: Clusters.csv not found at {CLUSTERS_CSV_PATH} for updating status.")
        return False

    rows = []
    updated = False
    fieldnames = []

    try:
        with open(CLUSTERS_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            if not fieldnames or CLUSTER_FIELD_COMPLETED not in fieldnames or CLUSTER_FIELD_PRIMARY_KEYWORD not in fieldnames:
                print(f"Error: CSV missing required headers ('{CLUSTER_FIELD_COMPLETED}', '{CLUSTER_FIELD_PRIMARY_KEYWORD}'). Found: {fieldnames}")
                return False
            
            for row in reader:
                if row.get(CLUSTER_FIELD_PRIMARY_KEYWORD) == cluster_primary_keyword:
                    row[CLUSTER_FIELD_COMPLETED] = new_status
                    updated = True
                    print(f"Marked cluster '{row.get(CLUSTER_FIELD_CLUSTER, cluster_primary_keyword)}' as '{new_status}'.")
                rows.append(row)
        
        if not updated:
            print(f"Warning: Cluster with Primary Keyword '{cluster_primary_keyword}' not found in {CLUSTERS_CSV_PATH}. No status updated.")
            return False # Or True if not finding it isn't an error for the caller

        # Write the modified data back to the CSV
        with open(CLUSTERS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Successfully updated {CLUSTERS_CSV_PATH}.")
        return True

    except Exception as e:
        print(f"Error updating {CLUSTERS_CSV_PATH}: {e}")
        return False

async def run_adk_agent_prompt(runner: Runner, session_id: str, user_id: str, prompt_text: str, agent_name_for_log: str) -> dict:
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=prompt_text)])
    agent_final_text = f"Agent {agent_name_for_log} did not produce a final response."
    raw_json_resp = None 
    error_message_str = None

    print(f"\nRunning ADK Agent prompt for '{agent_name_for_log}'...")
    try:
        async for event in runner.run_async(session_id=session_id, user_id=user_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    agent_final_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate and event.error_message:
                    escalation_error_msg = event.error_message 
                    agent_final_text = f"Agent '{agent_name_for_log}' escalated with error: {escalation_error_msg}"
                    error_message_str = agent_final_text
                    print(f"Error: {agent_final_text}")
                else:
                    agent_final_text = f"Agent {agent_name_for_log} produced a final response event with no parsable content or error message."
                    print(f"Warning: {agent_final_text} Event: {event}")
                break 
        
        if agent_final_text == f"Agent {agent_name_for_log} did not produce a final response." and not error_message_str:
             final_error_msg = f"Agent '{agent_name_for_log}' did not produce final text after iterating events."
             print(f"Error: {final_error_msg}")
             error_message_str = final_error_msg
    except Exception as e:
        exception_error_msg = f"Error during ADK call for {agent_name_for_log}: {e}"
        print(f"Error: {exception_error_msg}")
        agent_final_text = exception_error_msg
        error_message_str = exception_error_msg
        
    return {"text_content": agent_final_text, "raw_json_response": raw_json_resp, "error": error_message_str}

def get_internal_linking_data() -> str:
    """
    Reads Posted.csv (expecting Topic, Keywords, Summary, URL, Analysed cols)
    and returns a string formatted for the internal linking agent's prompt,
    including only posts marked 'Yes' as Analysed.
    """
    internal_links_text_parts = []
    if not os.path.exists(POSTED_CSV_PATH):
        print(f"Warning: {POSTED_CSV_PATH} not found. No internal links will be provided to the agent.")
        return "No internal linking data available (file not found)."

    try:
        with open(POSTED_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_posted_headers = ["Topic", "Keywords", "Summary", "URL", "Analysed"]
            if not reader.fieldnames or not all(h in reader.fieldnames for h in required_posted_headers):
                print(f"Warning: {POSTED_CSV_PATH} is missing one or more required headers for internal linking. Expected: {required_posted_headers}. Found: {reader.fieldnames}. No internal links will be used.")
                return "No internal linking data available (CSV header mismatch)."

            for row in reader:
                analysed_status = row.get("Analysed", "").strip().lower()
                url = row.get("URL", "").strip()
                if analysed_status == 'yes' and url:
                    topic = row.get("Topic", "N/A")
                    keywords = row.get("Keywords", "N/A")
                    summary = row.get("Summary", "N/A").replace('\n', ' ') # Consolidate summary for prompt
                    internal_links_text_parts.append(
                        f"- URL: {url}\n  Topic: {topic}\n  Keywords: {keywords}\n  Summary: {summary}\n"
                    )
        
        if not internal_links_text_parts:
            return "No suitable internal links found in Posted.csv (no posts marked 'Yes' or file is empty)."
        
        return "\n".join(internal_links_text_parts)
        
    except Exception as e:
        print(f"Error reading or processing {POSTED_CSV_PATH} for internal linking: {e}")
        return f"Error accessing internal linking data: {e}"

# --- Main Logic ---
async def main():
    print("Starting Blog Post Generation Process...")

    global CLUSTERS_CSV_PATH, POSTED_CSV_PATH, BLOG_OUTPUT_DIR
    
    # --- Path setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # All scripts and CSVs are in the 'seo_automation' folder (the script's directory).
    # Paths are relative to the script's location.
    CLUSTERS_CSV_PATH = os.path.join(script_dir, "Clusters.csv")
    # POSTED_CSV_FILENAME is defined globally
    POSTED_CSV_PATH = os.path.join(script_dir, POSTED_CSV_FILENAME) 
    BLOG_OUTPUT_DIR = os.path.join(script_dir, "generated_blog_posts")

    print(f"Script execution directory: {os.path.abspath(script_dir)}")
    print(f"Using Clusters CSV: {os.path.abspath(CLUSTERS_CSV_PATH)}")
    print(f"Using Posted URLs CSV: {os.path.abspath(POSTED_CSV_PATH)}")
    print(f"Blog post HTML output directory: {os.path.abspath(BLOG_OUTPUT_DIR)}")

    # Ensure BLOG_OUTPUT_DIR exists
    if not os.path.exists(BLOG_OUTPUT_DIR):
        try:
            os.makedirs(BLOG_OUTPUT_DIR)
            print(f"Created blog output directory: {os.path.abspath(BLOG_OUTPUT_DIR)}")
        except OSError as e:
            print(f"Error creating blog output directory {BLOG_OUTPUT_DIR}: {e}. Fallback: will attempt to save in script directory: {script_dir}")
            BLOG_OUTPUT_DIR = script_dir # Fallback

    # Ensure Clusters.csv exists (basic check, get_next_cluster handles more)
    if not os.path.exists(CLUSTERS_CSV_PATH):
        print(f"CRITICAL: Clusters.csv not found at {os.path.abspath(CLUSTERS_CSV_PATH)}. Please ensure it exists. Exiting.")
        return

    cluster_to_process = get_next_cluster_to_process()
    if not cluster_to_process:
        print("No suitable cluster found in Clusters.csv to process. Exiting.")
        return

    # --- Agent 1: Preliminary Planner (Gemini via OpenRouter) ---
    print("\n--- Step 1: Preliminary Blog Post Planning ---")
    prelim_planner_model = LiteLlm(
        model="openrouter/" + PRELIM_PLAN_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
    )
    preliminary_planner_agent = Agent(
        name="preliminary_blog_planner_agent",
        model=prelim_planner_model,
        instruction="", 
        tools=[],
    )
    session_service_agent1 = InMemorySessionService()
    artifact_service_agent1 = InMemoryArtifactService()
    session_agent1_id = "blog_post_gen_session_agent1"
    user_id_agent1 = "blog_writer_user_agent1"
    session_service_agent1.create_session(session_id=session_agent1_id, user_id=user_id_agent1, app_name='blog_post_generator_app_agent1')
    runner_agent1 = Runner(
        app_name='blog_post_generator_app_agent1',
        agent=preliminary_planner_agent,
        session_service=session_service_agent1,
        artifact_service=artifact_service_agent1
    )
    prompt_for_agent1 = PROMPT_AGENT_1_PRELIMINARY_PLAN.format(
        keywords=cluster_to_process.get(CLUSTER_FIELD_KEYWORDS, ""),
        intent=cluster_to_process.get(CLUSTER_FIELD_INTENT, ""),
        primary_keyword=cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD, "")
    )
    agent1_output_struct = await run_adk_agent_prompt(
        runner_agent1, session_agent1_id, user_id_agent1, prompt_for_agent1, "PreliminaryPlanner"
    )
    preliminary_plan_output = agent1_output_struct["text_content"]
    agent1_error = agent1_output_struct["error"]
    print("\nPreliminary Plan Output (raw):")
    print(preliminary_plan_output)
    if agent1_error or "did not produce a final response" in preliminary_plan_output or "escalated with error" in preliminary_plan_output:
        print(f"Failed to get preliminary plan: {agent1_error or preliminary_plan_output}. Exiting.")
        return

    # Update cluster status to 'Yes' immediately after successful Agent 1 processing
    processed_primary_keyword_for_status_update = cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD)
    if processed_primary_keyword_for_status_update:
        if update_cluster_status(processed_primary_keyword_for_status_update, "Yes"):
            print(f"Successfully updated status to 'Yes' for cluster with primary keyword: {processed_primary_keyword_for_status_update} after Agent 1.")
        else:
            print(f"Failed to update status for cluster with primary keyword: {processed_primary_keyword_for_status_update} after Agent 1.")
            # Decide if we should exit if status update fails. For now, let's print a warning and continue.
            print(f"Warning: Continuing blog generation despite failure to update CSV status for {processed_primary_keyword_for_status_update}.")
    else:
        print("Warning: Could not update cluster status after Agent 1 as primary keyword was not found in the processed cluster data.")

    # --- Step 2: Research - Direct LiteLLM Call ---
    print("\n--- Step 2: Researching Topic via Direct LiteLLM Call ---")
    research_query_for_direct_call = preliminary_plan_output 
    if not research_query_for_direct_call or research_query_for_direct_call.strip() == "":
        print("Preliminary plan output is empty. Using primary keyword for research instead.")
        research_query_for_direct_call = f"Provide comprehensive research information about: {cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD, 'general topic')}"
    else:
        research_query_for_direct_call = ' '.join(research_query_for_direct_call.splitlines()).replace('"', ' ')
        print(f"Using this preliminary plan as input for direct research call: '{research_query_for_direct_call[:200]}...'")
    
    research_result_struct = await fetch_and_process_research_directly(research_query_for_direct_call)
    research_findings_output = research_result_struct["text_content"]
    research_error = research_result_struct["error"]

    print("\nResearch Findings Output (from direct call, potentially with fixed links):")
    print(research_findings_output)

    if research_error or "Research failed" in research_findings_output:
        print(f"Critical error in research findings from direct LiteLLM call: {research_error or research_findings_output}. Exiting.")
        return

    # --- Agent 3: Detailed Plan Generation (ADK Agent) ---
    print("\n--- Step 3: Generating Detailed Blog Post Plan ---")
    detailed_plan_model = LiteLlm(
        model="openrouter/" + DETAILED_PLAN_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
    )
    detailed_plan_agent = Agent(
        name="detailed_blog_planner_agent",
        model=detailed_plan_model,
        instruction="", 
        tools=[],
    )
    session_service_agent3 = InMemorySessionService()
    artifact_service_agent3 = InMemoryArtifactService()
    session_agent3_id = "blog_post_gen_session_agent3"
    user_id_agent3 = "blog_writer_user_agent3"
    session_service_agent3.create_session(session_id=session_agent3_id, user_id=user_id_agent3, app_name='blog_post_generator_app_agent3')
    runner_agent3 = Runner(
        app_name='blog_post_generator_app_agent3',
        agent=detailed_plan_agent,
        session_service=session_service_agent3,
        artifact_service=artifact_service_agent3
    )
    prompt_for_agent3 = PROMPT_AGENT_3_DETAILED_PLAN.format(
        keywords=cluster_to_process.get(CLUSTER_FIELD_KEYWORDS, ""),
        intent=cluster_to_process.get(CLUSTER_FIELD_INTENT, ""),
        primary_keyword=cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD, ""),
        preliminary_plan=preliminary_plan_output,
        research_findings=research_findings_output 
    )
    agent3_output_struct = await run_adk_agent_prompt(
        runner_agent3, session_agent3_id, user_id_agent3, prompt_for_agent3, "DetailedPlanner"
    )
    detailed_plan_output = agent3_output_struct["text_content"] 
    agent3_error = agent3_output_struct["error"]
    print("\nDetailed Plan Output (raw):")
    print(detailed_plan_output)
    if agent3_error or "did not produce a final response" in detailed_plan_output or "escalated with error" in detailed_plan_output: 
        print(f"Failed to generate detailed plan: {agent3_error or detailed_plan_output}. Exiting further processing.")
        return
    
    print("\nBlog Post Generation Process (up to detailed plan) Finished.")

    # --- Agent 4: Write Blog Post (ADK Agent) ---
    print("\n--- Step 4: Writing Blog Post ---")
    write_blog_model = LiteLlm(
        model="openrouter/" + WRITE_BLOG_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
    )
    write_blog_agent = Agent(
        name="blog_writer_agent",
        model=write_blog_model,
        instruction="", 
        tools=[],
    )
    session_service_agent4 = InMemorySessionService()
    artifact_service_agent4 = InMemoryArtifactService()
    session_agent4_id = "blog_post_gen_session_agent4"
    user_id_agent4 = "blog_writer_user_agent4"
    session_service_agent4.create_session(session_id=session_agent4_id, user_id=user_id_agent4, app_name='blog_post_generator_app_agent4')
    runner_agent4 = Runner(
        app_name='blog_post_generator_app_agent4',
        agent=write_blog_agent,
        session_service=session_service_agent4,
        artifact_service=artifact_service_agent4
    )
    prompt_for_agent4 = PROMPT_AGENT_4_WRITE_BLOG.format(
        detailed_plan=detailed_plan_output,
        research_findings=research_findings_output,
        primary_keyword=cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD, "")
    )
    agent4_output_struct = await run_adk_agent_prompt(
        runner_agent4, session_agent4_id, user_id_agent4, prompt_for_agent4, "BlogWriter"
    )
    blog_post_output = agent4_output_struct["text_content"]
    agent4_error = agent4_output_struct["error"]
    print("\nGenerated Blog Post (raw):")
    print(blog_post_output)
    if agent4_error or "did not produce a final response" in blog_post_output or "escalated with error" in blog_post_output :
        print(f"Failed to generate blog post: {agent4_error or blog_post_output}")
        return
        
    # --- Agent 5: Add Internal Links ---
    print("\n--- Step 5: Adding Internal Links ---")
    internal_link_model = LiteLlm(
        model="openrouter/" + INTERNAL_LINK_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
    )
    internal_linker_agent = Agent(
        name="internal_linker_agent",
        model=internal_link_model,
        instruction="", # The main instruction is in the dynamic prompt
        tools=[],
    )
    session_service_agent5 = InMemorySessionService() # New session service for this agent
    artifact_service_agent5 = InMemoryArtifactService()
    session_agent5_id = f"blog_post_gen_session_agent5_{cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD, 'default_pk').replace(' ','_')}"
    user_id_agent5 = "blog_writer_user_agent5"
    session_service_agent5.create_session(session_id=session_agent5_id, user_id=user_id_agent5, app_name='blog_post_generator_app_agent5')
    
    runner_agent5 = Runner(
        app_name='blog_post_generator_app_agent5',
        agent=internal_linker_agent,
        session_service=session_service_agent5,
        artifact_service=artifact_service_agent5
    )

    internal_links_str_data = get_internal_linking_data()
    
    prompt_for_agent5 = PROMPT_AGENT_5_INTERNAL_LINKS.format(
        current_blog_post_content=blog_post_output,
        internal_linking_data=internal_links_str_data
    )
    
    agent5_output_struct = await run_adk_agent_prompt(
        runner_agent5, session_agent5_id, user_id_agent5, prompt_for_agent5, "InternalLinker"
    )
    blog_post_with_internal_links = agent5_output_struct["text_content"]
    agent5_error = agent5_output_struct["error"]

    print("\nBlog Post with Internal Links (raw):")
    print(blog_post_with_internal_links[:1000] + "..." if len(blog_post_with_internal_links) > 1000 else blog_post_with_internal_links) # Print snippet

    if agent5_error or "did not produce a final response" in blog_post_with_internal_links or "escalated with error" in blog_post_with_internal_links:
        print(f"Failed to add internal links: {agent5_error or blog_post_with_internal_links}. Proceeding with content from Agent 4 for HTML conversion.")
        # Fallback to Agent 4's output if Agent 5 fails
        content_for_html_conversion = blog_post_output 
    else:
        content_for_html_conversion = blog_post_with_internal_links

    # --- Agent 6: Convert to HTML ---
    print("\n--- Step 6: Converting to HTML ---")
    html_model = LiteLlm(
        model="openrouter/" + HTML_CONVERSION_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
    )
    html_converter_agent = Agent(
        name="html_converter_agent",
        model=html_model,
        instruction="",
        tools=[],
    )
    session_service_agent6 = InMemorySessionService()
    artifact_service_agent6 = InMemoryArtifactService()
    session_agent6_id = f"blog_post_gen_session_agent6_{cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD, 'default_pk').replace(' ','_')}"
    user_id_agent6 = "blog_writer_user_agent6"
    session_service_agent6.create_session(session_id=session_agent6_id, user_id=user_id_agent6, app_name='blog_post_generator_app_agent6')

    runner_agent6 = Runner(
        app_name='blog_post_generator_app_agent6',
        agent=html_converter_agent,
        session_service=session_service_agent6,
        artifact_service=artifact_service_agent6
    )

    prompt_for_agent6 = PROMPT_AGENT_6_HTML_CONVERSION.format(
        final_blog_post_content_with_links=content_for_html_conversion
    )

    agent6_output_struct = await run_adk_agent_prompt(
        runner_agent6, session_agent6_id, user_id_agent6, prompt_for_agent6, "HtmlConverter"
    )
    final_html_output = agent6_output_struct["text_content"]
    agent6_error = agent6_output_struct["error"]

    print("\nFinal HTML Output (raw snippet):")
    print(final_html_output[:1000] + "..." if len(final_html_output) > 1000 else final_html_output) # Print snippet

    if agent6_error or "did not produce a final response" in final_html_output or "escalated with error" in final_html_output:
        print(f"Failed to convert blog post to HTML: {agent6_error or final_html_output}. No HTML file will be saved.")
    else:
        # Save the HTML to a file
        primary_keyword_for_filename = cluster_to_process.get(CLUSTER_FIELD_PRIMARY_KEYWORD, "untitled_blog_post")

        sanitized_filename = re.sub(r'[^a-zA-Z0-9_\-]+', '', primary_keyword_for_filename.replace(' ', '_')) + ".html"
        output_html_path = os.path.join(BLOG_OUTPUT_DIR, sanitized_filename)
        try:
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(final_html_output)
            print(f"Successfully saved HTML to: {os.path.abspath(output_html_path)}")
        except Exception as e:
            print(f"Error saving HTML file to {output_html_path}: {e}")
            
    print("\nBlog Post Generation Process (including HTML) Fully Finished.")

if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("CRITICAL: OPENROUTER_API_KEY environment variable is not set. This is required for all agents. Exiting.")
        exit(1)
        
    print("Running blog post generator manually...")
    asyncio.run(main()) 