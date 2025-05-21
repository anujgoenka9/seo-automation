import os
import asyncio
import csv
import re
from collections import Counter
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
    print("Error: OPENROUTER_API_KEY not found in environment variables. Please set it in a .env file.")
    exit(1)

BASE_FILE_PATH = "."
COMPETITOR_ANALYSIS_CSV_PATH = os.path.join(BASE_FILE_PATH, "Competitor Analysis.csv")
CLUSTERS_CSV_PATH = os.path.join(BASE_FILE_PATH, "Clusters.csv")

CLUSTER_CSV_HEADERS = ["Cluster", "Intent", "Keywords", "Primary Keyword", "Completed"]

KEYWORD_MODEL_NAME = "google/gemini-2.5-flash-preview:thinking"

# --- Prompt Definitions ---

PROMPT_1_PILLAR_CLUSTER_GENERATION = """
You are part of a team that provides SEO services for clients. In order to boost SEO, your team posts SEO optimised blogs weekly which target high volume keywords related to the clients niche. The goal is to rank for keywords and concepts which are related to the clients products and services, so when their target market is looking for their service on Google they will show up first.
Your job is to come up with the keywords your team targets in their blogs. The SEO strategy your team follows is to have multiple clusters of keywords. Each cluster is a group of keywords which are related to one another and will likely go into the same blog post. Then multiple related clusters will go under one pillar post. The pillar post will have links to all of the posts from the clusters in its group. So at a high level, the clusters go into detail on specific topics whilst the pillar post is more high level and just touches on all the topics in its clusters. Therefore, to find the best keywords, clusters and pillar posts you:
- First start with the products and services and think of the ICP.
- Then identify all the potential searches your clients customers would do on Google. Think of FAQs..
- Then identify keywords for each search intent which are high volume and highly relevant
(SEO optimised).
- Then create clusters.
- Then create the pillar posts.
Your output must follow this format:
# Pillar Post 1
## Cluster 1 keywords (dot point format)
## Cluster 2 keywords (dot point format)
..
# Pillar Post 2
## Cluster 1 keywords (dot point format)
## Cluster 2 keywords (dot point format)
..
[Continue]
You have just signed on a new client. The client's name is San Diego Dental Studio and they provide preventive care, dental restorations and cosmetic dentistry to clients.
Create the keyword plan in desired format.
"""

# Prompt 2 will be constructed dynamically with output from Prompt 1 and competitor keywords.

# --- Helper Functions ---

def get_top_competitor_keywords(top_n=5) -> list[str]:
    """Reads Competitor Analysis.csv and returns the top N most frequent unique keywords."""
    all_keywords_flat = []
    try:
        with open(COMPETITOR_ANALYSIS_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if "Keywords" not in reader.fieldnames:
                print(f"Warning: 'Keywords' column not found in {COMPETITOR_ANALYSIS_CSV_PATH}. Cannot extract competitor keywords.")
                return []
            for row in reader:
                keywords_str = row.get("Keywords", "")
                if keywords_str:
                    keywords_list = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                    all_keywords_flat.extend(keywords_list)

        if not all_keywords_flat:
            print(f"No keywords found in {COMPETITOR_ANALYSIS_CSV_PATH}.")
            return []

        keyword_counts = Counter(all_keywords_flat)
        most_common_keywords = [kw for kw, count in keyword_counts.most_common(top_n)]
        print(f"Top {len(most_common_keywords)} competitor keywords found: {most_common_keywords}")
        return most_common_keywords

    except FileNotFoundError:
        print(f"Error: {COMPETITOR_ANALYSIS_CSV_PATH} not found. Cannot extract competitor keywords.")
        return []
    except Exception as e:
        print(f"Error reading competitor keywords from {COMPETITOR_ANALYSIS_CSV_PATH}: {e}")
        return []

def initialize_clusters_csv():
    """Ensures Clusters.csv exists with the correct headers."""
    if not os.path.exists(CLUSTERS_CSV_PATH):
        print(f"Creating {CLUSTERS_CSV_PATH} with headers.")
        try:
            with open(CLUSTERS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(CLUSTER_CSV_HEADERS)
        except IOError as e:
            print(f"Error creating CSV file '{CLUSTERS_CSV_PATH}': {e}")
            raise
    else:
        print(f"{CLUSTERS_CSV_PATH} already exists.")

def write_clusters_to_csv(cluster_data_list: list[dict]):
    """Appends a list of cluster data dictionaries to Clusters.csv."""
    if not cluster_data_list:
        print("No cluster data to write.")
        return

    try:
        with open(CLUSTERS_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CLUSTER_CSV_HEADERS)
            if os.path.getsize(CLUSTERS_CSV_PATH) <= len(",".join(CLUSTER_CSV_HEADERS)) + 5:
                 pass

            for cluster_dict in cluster_data_list:
                row_to_write = {header: cluster_dict.get(header, "") for header in CLUSTER_CSV_HEADERS}
                if "Completed" not in cluster_dict:
                    row_to_write["Completed"] = ""
                writer.writerow(row_to_write)
        print(f"Successfully wrote {len(cluster_data_list)} rows to {CLUSTERS_CSV_PATH}")
    except IOError as e:
        print(f"Error writing to {CLUSTERS_CSV_PATH}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing to {CLUSTERS_CSV_PATH}: {e}")

def parse_llm_table_output(markdown_table: str) -> list[dict]:
    """Parses the LLM's markdown table output into a list of dictionaries."""
    parsed_data = []
    if not markdown_table or not markdown_table.strip():
        print("Warning: Empty or whitespace-only markdown_table received from LLM for parsing.")
        return []

    lines = [line.strip() for line in markdown_table.strip().split('\n') if line.strip()]
    
    header_idx = -1
    separator_idx = -1
    
    for i, line in enumerate(lines):
        if line.startswith('|') and line.endswith('|'):
            if "---" in line:
                if header_idx == i -1:
                    separator_idx = i
                    break 
            elif header_idx == -1:
                header_idx = i
                
    if header_idx == -1 or separator_idx == -1:
        print(f"Warning: Could not find valid markdown table header and separator. Table:\n{markdown_table}")
        return []

    expected_llm_headers = CLUSTER_CSV_HEADERS[:-1]

    for i in range(separator_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith('|') and line.endswith('|'):
            cols = [col.strip() for col in line.split('|')[1:-1]]
            if len(cols) == len(expected_llm_headers):
                row_dict = dict(zip(expected_llm_headers, cols))
                row_dict["Completed"] = ""
                parsed_data.append(row_dict)
            else:
                print(f"Warning: Skipping malformed table row. Expected {len(expected_llm_headers)} cols, got {len(cols)}. Row: '{line}'")
    
    if not parsed_data:
        print(f"Warning: No data rows extracted from table. Table:\n{markdown_table}")
        
    return parsed_data

async def run_llm_prompt(runner: Runner, session_id: str, full_prompt_text: str, agent_name: str) -> str:
    """Helper to run a full prompt (instructions + query) against the specified agent and get the final text response."""
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=full_prompt_text)])
    agent_final_text = ""
    print(f"\nRunning prompt for agent '{agent_name}'...")
    
    try:
        async for event in runner.run_async(session_id=session_id, user_id='keyword_planner_user', new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    agent_final_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate and event.error_message:
                    agent_final_text = f"Agent escalated with error: {event.error_message}"
                    print(f"Error: Agent '{agent_name}' escalated: {event.error_message}")
                else:
                    print(f"Warning: Final response from agent '{agent_name}' had no parsable content. Event: {event}")
                break
        if not agent_final_text:
            print(f"Error: Agent '{agent_name}' did not produce final text.")
    except Exception as e:
        print(f"Error running LLM prompt for agent '{agent_name}': {e}")
        agent_final_text = f"Error during LLM call: {e}"
    return agent_final_text

# --- Main Logic ---
async def main():
    print("Starting Keyword Planning Process...")

    global COMPETITOR_ANALYSIS_CSV_PATH, CLUSTERS_CSV_PATH
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    COMPETITOR_ANALYSIS_CSV_PATH = os.path.join(script_dir, "Competitor Analysis.csv")
    CLUSTERS_CSV_PATH = os.path.join(script_dir, "Clusters.csv")

    print(f"Script execution directory: {os.path.abspath(script_dir)}")
    print(f"Using Competitor Analysis CSV: {os.path.abspath(COMPETITOR_ANALYSIS_CSV_PATH)}")
    print(f"Using Clusters Output CSV: {os.path.abspath(CLUSTERS_CSV_PATH)}")

    initialize_clusters_csv()
    
    top_keywords = get_top_competitor_keywords(top_n=5)

    keyword_model = LiteLlm(
        model="openrouter/" + KEYWORD_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
    )
    
    planner_agent = Agent(
        name="seo_keyword_planner_agent",
        model=keyword_model,
        instruction="",
        tools=[],
    )

    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session = session_service.create_session(user_id='keyword_planner_user', app_name='keyword_planner_app')
    runner = Runner(
        app_name='keyword_planner_app',
        agent=planner_agent,
        session_service=session_service,
        artifact_service=artifact_service
    )

    print("\n--- Step 1: Generating Initial Pillar & Cluster Ideas ---")
    pillar_cluster_output = await run_llm_prompt(runner, session.id, PROMPT_1_PILLAR_CLUSTER_GENERATION, "PillarClusterGenerator")

    if not pillar_cluster_output or "Error during LLM call" in pillar_cluster_output or "Agent escalated" in pillar_cluster_output:
        print("Failed to get initial pillar/cluster ideas. Exiting.")
        return

    print("\nInitial Pillar/Cluster Ideas Received (raw):")
    print(pillar_cluster_output)

    print("\n--- Step 2: Refining with Competitor Keywords & Formatting to Table ---")
    
    competitor_keywords_str = ", ".join(top_keywords) if top_keywords else "No specific competitor keywords available."
    
    PROMPT_2_REFINE_AND_TABLE = f"""
You have previously generated the following keyword plan:
--- BEGIN INITIAL PLAN ---
{pillar_cluster_output}
--- END INITIAL PLAN ---

Additionally, our top competitor, who currently ranks highly, is focusing on these keywords:
Competitor Keywords: {competitor_keywords_str}
This information is crucial because understanding what top competitors are targeting helps refine our own strategy to find gaps or areas to also cover.

Now, carefully review the initial plan. If necessary, refine it by thoughtfully incorporating insights or themes from these competitor keywords.
The primary goal is to produce a comprehensive and effective keyword strategy.

After any necessary refinement, your FINAL output MUST be formatted ONLY as a markdown table with the following columns:
- Cluster: Cluster name
- Intent: Search intent of the user from searching for these keywords. This would go to define the topic of the blog post.
- Keywords: Keywords to target in that cluster (comma-separated).
- Primary Keyword: Most important keyword of the list of keywords in this cluster. This would be used to create the title of the blog post.

Here's an example of the required table format:
| Cluster                       | Intent                                                                                                                    | Keywords                                                                                                | Primary Keyword         |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-------------------------|
| Cosmetic Dentistry Options    | Users looking for ways to improve the appearance of their smile, exploring different cosmetic procedures.                 | smile makeover, teeth whitening, dental veneers, cosmetic bonding, best cosmetic dentist                 | cosmetic dentistry      |
| Emergency Dental Care         | Users experiencing urgent dental problems (toothache, broken tooth) seeking immediate help.                                 | emergency dentist, urgent dental care, broken tooth repair, severe toothache relief, same day dentist | emergency dentist       |

Do NOT output any text before or after this single markdown table.
The table should contain the finalized cluster information.
"""

    final_table_output = await run_llm_prompt(runner, session.id, PROMPT_2_REFINE_AND_TABLE, "TableFormatter")

    if not final_table_output or "Error during LLM call" in final_table_output or "Agent escalated" in final_table_output:
        print("Failed to get formatted table output. Exiting.")
        return

    print("\nFinal Formatted Table Received (raw):")
    print(final_table_output)

    print("\n--- Step 3: Parsing Table and Writing to CSV ---")
    parsed_clusters = parse_llm_table_output(final_table_output)

    if parsed_clusters:
        write_clusters_to_csv(parsed_clusters)
    else:
        print("No clusters parsed from the LLM output. CSV not updated.")
        print("Please check the raw output from 'TableFormatter' above to see if it provided a valid table.")

    print("\nKeyword Planning Process Finished.")

if __name__ == "__main__":
    asyncio.run(main()) 