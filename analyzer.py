import os
import re
import time
import asyncio
import csv
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.genai import types as genai_types

# --- Configuration ---
COMPETITOR_URLS_SHEET_NAME = "Competitor URLs"
ANALYSIS_OUTPUT_SHEET_NAME = "Competitor Analysis"

BASE_FILE_PATH = "."
COMPETITOR_URLS_CSV_PATH = os.path.join(BASE_FILE_PATH, f"{COMPETITOR_URLS_SHEET_NAME}.csv")
ANALYSIS_OUTPUT_CSV_PATH = os.path.join(BASE_FILE_PATH, f"{ANALYSIS_OUTPUT_SHEET_NAME}.csv")
POSTED_CSV_FILENAME = "Posted.csv"
POSTED_CSV_PATH = os.path.join(BASE_FILE_PATH, POSTED_CSV_FILENAME)

URL_COL_COMP_SHEET = "URL"
ANALYSED_COL_COMP_SHEET = "Analysed"
TOPIC_COL_ANALYSIS_SHEET = "Topic"
KEYWORDS_COL_ANALYSIS_SHEET = "Keywords"
SUMMARY_COL_ANALYSIS_SHEET = "Summary"
URL_COL_ANALYSIS_SHEET = "URL"

POSTED_CSV_HEADERS = [TOPIC_COL_ANALYSIS_SHEET, KEYWORDS_COL_ANALYSIS_SHEET, SUMMARY_COL_ANALYSIS_SHEET, URL_COL_ANALYSIS_SHEET, ANALYSED_COL_COMP_SHEET]

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in environment variables. Please set it in a .env file.")
    exit(1)

# --- LLM and Agent Setup ---
model = LiteLlm(
    model="openrouter/openai/gpt-4o-mini-search-preview",
    api_key=OPENROUTER_API_KEY,
)

AGENT_INSTRUCTION = """
You are part of a professional SEO team.
Your job is to analyse competitors' blog posts to help your team come up with a content and keyword strategy.
You will be given a blog post URL directly in the prompt. You should access and analyze the content of this URL yourself.
Based on the content you retrieve from the URL, identify the following:
- Topic: The topic of the blog post.
- Keywords: The 3 top SEO keywords in the blog which are related to the competitor's line of business. Identify a blend of long-tail and short-tail keywords. The keyword must be directly present in the competitor's blog.
- Summary: A concise dot-point summary of what the blog post was about. This should be short, and every dot point should capture a different subtopic within the blog. Use <br> to separate distinct dot points within this single 'Summary' cell.

The competitor is a Dental company with the following services (lines of business):
preventative, restorative, and cosmetic dentistry

Format your output ONLY as a markdown table with the following columns: Topic, Keywords, Summary.
The table MUST have exactly three rows: a header row, a separator row, and ONE SINGLE data row containing the analysis.
The data row must strictly follow the format: | Topic Value | Keywords Value | Summary Value (with <br> for internal newlines) |

Example of the expected output format:
| Topic | Keywords | Summary |
|------------------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Symptoms Indicating the Need for Wisdom Teeth Removal | wisdom teeth removal, impacted wisdom teeth, dental pain, swollen gums, jaw stiffness | - Defines wisdom teeth as third molars that emerge in late teens or early twenties.<br>- Explains common issues like impaction due to lack of space.<br>- Highlights symptoms such as persistent pain, jaw stiffness, and swollen gums.<br>- Describes pericoronitis as an infection from trapped food and bacteria.<br>- Recommends consulting a dentist if experiencing any of these signs. |

Do not output any introductory text, explanations, or anything else besides the single markdown table with its three required rows (header, separator, one data row). Ensure the keywords are directly from the blog content.
"""

analyzer_agent = Agent(
    name="seo_competitor_analyzer",
    model=model,
    instruction=AGENT_INSTRUCTION,
    tools=[],
)

# --- CSV Helper Functions ---
def _ensure_csv_with_headers(file_path, expected_headers):
    if not os.path.exists(file_path):
        print(f"CSV file '{file_path}' not found. Creating with headers.")
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(expected_headers)
            print(f"Created CSV '{file_path}' with headers.")
        except IOError as e:
            print(f"Error creating CSV file '{file_path}': {e}")
            raise
    else:
        print(f"CSV file '{file_path}' already exists.")

def initialize_csv_files():
    print(f"Ensuring CSV file: {COMPETITOR_URLS_CSV_PATH}")
    _ensure_csv_with_headers(COMPETITOR_URLS_CSV_PATH, [URL_COL_COMP_SHEET, ANALYSED_COL_COMP_SHEET])
    print(f"Ensuring CSV file: {ANALYSIS_OUTPUT_CSV_PATH}")
    _ensure_csv_with_headers(ANALYSIS_OUTPUT_CSV_PATH, [TOPIC_COL_ANALYSIS_SHEET, KEYWORDS_COL_ANALYSIS_SHEET, SUMMARY_COL_ANALYSIS_SHEET, URL_COL_ANALYSIS_SHEET])
    print(f"Ensuring CSV file: {POSTED_CSV_PATH}")
    _ensure_csv_with_headers(POSTED_CSV_PATH, POSTED_CSV_HEADERS)

def get_urls_to_analyze_csv():
    urls_data = []
    try:
        with open(COMPETITOR_URLS_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if URL_COL_COMP_SHEET not in reader.fieldnames or ANALYSED_COL_COMP_SHEET not in reader.fieldnames:
                print(f"Error: Missing required columns ('{URL_COL_COMP_SHEET}' or '{ANALYSED_COL_COMP_SHEET}') in '{COMPETITOR_URLS_CSV_PATH}'.")
                return []
            for i, row in enumerate(reader):
                url = row.get(URL_COL_COMP_SHEET, "").strip()
                analysed_status = row.get(ANALYSED_COL_COMP_SHEET, "").strip().lower()
                if url and analysed_status == "no":
                    urls_data.append({"url": url, "original_row_index": i + 2})
    except FileNotFoundError:
        print(f"Error: '{COMPETITOR_URLS_CSV_PATH}' not found. Please create it or run the converter script.")
        initialize_csv_files()
        return []
    except Exception as e:
        print(f"Error reading '{COMPETITOR_URLS_CSV_PATH}': {e}")
        return []
    return urls_data

def get_posted_urls_to_analyze():
    urls_data = []
    try:
        with open(POSTED_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not all(col in reader.fieldnames for col in [URL_COL_ANALYSIS_SHEET, ANALYSED_COL_COMP_SHEET]):
                print(f"Error: Missing required columns ('{URL_COL_ANALYSIS_SHEET}' or '{ANALYSED_COL_COMP_SHEET}') in '{POSTED_CSV_PATH}'.")
                return []
            for i, row in enumerate(reader):
                url = row.get(URL_COL_ANALYSIS_SHEET, "").strip()
                analysed_status = row.get(ANALYSED_COL_COMP_SHEET, "").strip().lower()
                if url and (analysed_status == "no" or not analysed_status):
                    urls_data.append({"url": url})
    except FileNotFoundError:
        print(f"Error: '{POSTED_CSV_PATH}' not found. Please create it. Attempting to initialize.")
        initialize_csv_files()
        return []
    except Exception as e:
        print(f"Error reading '{POSTED_CSV_PATH}': {e}")
        return []
    return urls_data

def write_analysis_data_csv(data_dict):
    output_buffer = []
    url_to_update = data_dict[URL_COL_ANALYSIS_SHEET]
    updated_in_file = False

    canonical_fieldnames = [TOPIC_COL_ANALYSIS_SHEET, KEYWORDS_COL_ANALYSIS_SHEET, SUMMARY_COL_ANALYSIS_SHEET, URL_COL_ANALYSIS_SHEET]
    current_fieldnames = []

    try:
        with open(ANALYSIS_OUTPUT_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)

            if header and any(h.strip() for h in header):
                current_fieldnames = header
                output_buffer.append(current_fieldnames)

                url_col_idx = -1
                try:
                    url_col_idx = current_fieldnames.index(URL_COL_ANALYSIS_SHEET)
                except ValueError:
                    print(f"Warning: '{URL_COL_ANALYSIS_SHEET}' not in '{ANALYSIS_OUTPUT_CSV_PATH}' headers. Update/Append might be problematic.")

                for row_list in reader:
                    if not any(cell.strip() for cell in row_list):
                        continue

                    is_target_row = False
                    if url_col_idx != -1 and url_col_idx < len(row_list) and row_list[url_col_idx] == url_to_update:
                        is_target_row = True

                    if is_target_row:
                        print(f"Updating data for URL {url_to_update} in '{ANALYSIS_OUTPUT_CSV_PATH}'")
                        updated_row_values = []
                        temp_row_dict_for_update = dict(zip(current_fieldnames, row_list))
                        for fn_header in current_fieldnames:
                            updated_row_values.append(data_dict.get(fn_header, temp_row_dict_for_update.get(fn_header, "")))
                        output_buffer.append(updated_row_values)
                        updated_in_file = True
                    else:
                        output_buffer.append(row_list)
            else:
                print(f"'{ANALYSIS_OUTPUT_CSV_PATH}' is empty or has no valid header. Will create/overwrite with new data and canonical headers.")
                current_fieldnames = canonical_fieldnames
                output_buffer.append(current_fieldnames)

        if not updated_in_file:
            print(f"Appending new data for URL {url_to_update} to '{ANALYSIS_OUTPUT_CSV_PATH}'")
            if not output_buffer or not output_buffer[0]:
                header_for_append = current_fieldnames if current_fieldnames and any(h.strip() for h in current_fieldnames) else canonical_fieldnames
                if not output_buffer:
                    output_buffer.append(header_for_append)
                else:
                    output_buffer[0] = header_for_append
            else:
                header_for_append = output_buffer[0]

            new_row_values = [data_dict.get(fn, "") for fn in header_for_append]
            output_buffer.append(new_row_values)

        with open(ANALYSIS_OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_buffer)
        print(f"Successfully wrote to '{ANALYSIS_OUTPUT_CSV_PATH}' for URL {url_to_update}")

    except FileNotFoundError:
        print(f"File '{ANALYSIS_OUTPUT_CSV_PATH}' not found. Creating new file with data.")
        output_buffer.append(canonical_fieldnames)
        new_row_values = [data_dict.get(fn, "") for fn in canonical_fieldnames]
        output_buffer.append(new_row_values)
        with open(ANALYSIS_OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_buffer)
        print(f"Successfully created and wrote to '{ANALYSIS_OUTPUT_CSV_PATH}' for URL {url_to_update}")
    except Exception as e:
        print(f"Error writing to '{ANALYSIS_OUTPUT_CSV_PATH}': {e}")
        import traceback
        traceback.print_exc()

def update_posted_csv_data(url_to_update: str, analysis_results: dict):
    """
    Updates an existing row in Posted.csv with new analysis data and marks it as 'Yes'.
    analysis_results should be a dict with keys TOPIC_COL_ANALYSIS_SHEET, etc.
    """
    rows = []
    updated = False

    try:
        with open(POSTED_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                print(f"Error: {POSTED_CSV_PATH} seems to be empty or has no headers.")
                return

            if not all(col in reader.fieldnames for col in POSTED_CSV_HEADERS):
                print(f"Error: '{POSTED_CSV_PATH}' is missing some of the required headers: {POSTED_CSV_HEADERS}. Found: {reader.fieldnames}")
                return

            fieldnames = reader.fieldnames

            for row in reader:
                if row.get(URL_COL_ANALYSIS_SHEET) == url_to_update:
                    row[TOPIC_COL_ANALYSIS_SHEET] = analysis_results.get(TOPIC_COL_ANALYSIS_SHEET, row.get(TOPIC_COL_ANALYSIS_SHEET, ""))
                    row[KEYWORDS_COL_ANALYSIS_SHEET] = analysis_results.get(KEYWORDS_COL_ANALYSIS_SHEET, row.get(KEYWORDS_COL_ANALYSIS_SHEET, ""))
                    row[SUMMARY_COL_ANALYSIS_SHEET] = analysis_results.get(SUMMARY_COL_ANALYSIS_SHEET, row.get(SUMMARY_COL_ANALYSIS_SHEET, ""))
                    row[ANALYSED_COL_COMP_SHEET] = "Yes"
                    updated = True
                    print(f"Prepared update for URL {url_to_update} in '{POSTED_CSV_PATH}'")
                rows.append(row)

        if not updated:
            print(f"Warning: URL '{url_to_update}' not found in '{POSTED_CSV_PATH}' for updating. No changes made to this file for this URL.")
            return

        with open(POSTED_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Successfully updated data in '{POSTED_CSV_PATH}' for URL {url_to_update}")

    except FileNotFoundError:
        print(f"Error: '{POSTED_CSV_PATH}' not found during update. It should have been created.")
    except Exception as e:
        print(f"Error updating '{POSTED_CSV_PATH}' for URL {url_to_update}: {e}")
        import traceback
        traceback.print_exc()

def mark_url_as_analyzed_csv(url_info_to_mark):
    output_buffer = []
    url_to_mark = url_info_to_mark["url"]
    found_and_marked = False

    current_fieldnames = []

    try:
        with open(COMPETITOR_URLS_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)

            if header and any(h.strip() for h in header):
                current_fieldnames = header
                output_buffer.append(current_fieldnames)

                url_col_idx = -1
                analysed_col_idx = -1
                try:
                    url_col_idx = current_fieldnames.index(URL_COL_COMP_SHEET)
                    analysed_col_idx = current_fieldnames.index(ANALYSED_COL_COMP_SHEET)
                except ValueError:
                    print(f"Warning: Missing required columns ('{URL_COL_COMP_SHEET}' or '{ANALYSED_COL_COMP_SHEET}') in '{COMPETITOR_URLS_CSV_PATH}' header. Cannot mark URLs.")
                    for row_list in reader:
                        if not any(cell.strip() for cell in row_list):
                            continue
                        output_buffer.append(row_list)

                if url_col_idx != -1 and analysed_col_idx != -1:
                    for row_list in reader:
                        if not any(cell.strip() for cell in row_list):
                            continue

                        current_row_list_copy = list(row_list)

                        if url_col_idx < len(current_row_list_copy) and current_row_list_copy[url_col_idx] == url_to_mark:
                            if analysed_col_idx < len(current_row_list_copy):
                                current_row_list_copy[analysed_col_idx] = "Yes"
                                print(f"Marking URL {url_to_mark} as 'Yes' in '{COMPETITOR_URLS_CSV_PATH}'")
                                found_and_marked = True
                            else:
                                print(f"Warning: Row for {url_to_mark} in '{COMPETITOR_URLS_CSV_PATH}' is too short to mark 'Analysed'. Length: {len(current_row_list_copy)}, Expected index: {analysed_col_idx}")
                        output_buffer.append(current_row_list_copy)

            else:
                print(f"'{COMPETITOR_URLS_CSV_PATH}' is empty or has no valid header. Cannot mark URL or clean file.")
                return

        if not found_and_marked and (header and any(h.strip() for h in header) and URL_COL_COMP_SHEET in header):
            print(f"Warning: URL '{url_to_mark}' not found in '{COMPETITOR_URLS_CSV_PATH}' to mark as analyzed (after filtering empty rows).")

        if output_buffer:
            with open(COMPETITOR_URLS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(output_buffer)
            print(f"Successfully updated '{COMPETITOR_URLS_CSV_PATH}' (empty rows removed if any).")
        else:
            print(f"No content to write for '{COMPETITOR_URLS_CSV_PATH}' after processing.")

    except FileNotFoundError:
        print(f"Error: '{COMPETITOR_URLS_CSV_PATH}' not found during marking analyzed. It should have been created by initialize_csv_files.")
    except Exception as e:
        print(f"Error updating '{COMPETITOR_URLS_CSV_PATH}': {e}")


# --- Markdown Parsing Function ---
def parse_ai_table_output(markdown_table: str) -> dict | None:
    if not markdown_table or not markdown_table.strip():
        print("Warning: Empty or whitespace-only markdown_table received from LLM.")
        return None

    lines = [line.strip() for line in markdown_table.strip().split('\\n') if line.strip()]

    header_line_index = -1
    for i, line in enumerate(lines):
        if line.startswith('|') and line.endswith('|') and "---" not in line:
            header_line_index = i
            break

    if header_line_index == -1:
        print(f"Warning: Could not find a valid header row in table. Content:\\n{markdown_table}")
        return None

    raw_headers = [h.strip() for h in lines[header_line_index].split('|')[1:-1]]

    data_row_index = -1
    for i in range(header_line_index + 1, len(lines)):
        line_content = lines[i]
        if line_content.startswith('|') and \
           not re.match(r"^[|\\s:-]+$", line_content):
            data_row_index = i
            break

    if data_row_index == -1:
        print(f"Warning: Could not find a data row after header. Content:\\n{markdown_table}")
        return None

    raw_data_columns = [col.strip() for col in lines[data_row_index].split('|')[1:]]

    if len(raw_data_columns) >= 3:
        topic = raw_data_columns[0]
        keywords = raw_data_columns[1]
        summary = raw_data_columns[2].replace("<br>", "\\n").replace("<br/>", "\\n").replace("<br />", "\\n")

        if not topic and len(raw_data_columns[0]) == 0:
             print(f"Warning: Parsed 'Topic' is empty. Row: '{lines[data_row_index]}'")
        if not keywords and len(raw_data_columns[1]) == 0:
             print(f"Warning: Parsed 'Keywords' is empty. Row: '{lines[data_row_index]}'")
        if not summary and len(raw_data_columns[2]) == 0:
             print(f"Warning: Parsed 'Summary' is empty. Row: '{lines[data_row_index]}'")

        return {
            TOPIC_COL_ANALYSIS_SHEET: topic,
            KEYWORDS_COL_ANALYSIS_SHEET: keywords,
            SUMMARY_COL_ANALYSIS_SHEET: summary,
        }
    else:
        print(f"Warning: Expected at least 3 data columns based on prompt, found {len(raw_data_columns)}. Row: '{lines[data_row_index]}'. Table:\\n{markdown_table}")
        return None

# --- Main Logic ---
async def _run_agent_and_parse(runner: Runner, session_id: str, url_to_analyze: str) -> dict | None:
    """
    Helper function to run the ADK agent for a given URL and parse its output.
    Returns a dictionary with parsed data (Topic, Keywords, Summary) or None if failed.
    """
    print(f"Analyzing URL with ADK agent: {url_to_analyze}")
    query = f"Please analyze the following URL: {url_to_analyze}"
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])
    agent_final_text = ""
    parsed_data = None

    try:
        event_count = 0
        async for event in runner.run_async(session_id=session_id, user_id='analyzer_user', new_message=content):
            event_count += 1
            if event.is_final_response():
                if event.content and event.content.parts:
                    agent_final_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate and event.error_message:
                    agent_final_text = f"Agent escalated with error: {event.error_message}"
                    print(f"Error: Agent escalated for {url_to_analyze}: {event.error_message}")
                else:
                    print(f"Warning: Final response for {url_to_analyze} had no parsable content. Event: {event}")
                break

        if not agent_final_text:
            print(f"Error: Agent did not produce final text for {url_to_analyze} after {event_count} events.")
            return None

        parsed_data = parse_ai_table_output(agent_final_text)
        if not parsed_data:
            print(f"Failed to parse agent's output for {url_to_analyze}. Raw output was:\\n{agent_final_text}")
            return None

        return parsed_data

    except Exception as e:
        print(f"Error during agent processing for URL {url_to_analyze}: {e}")
        return None

async def process_single_competitor_url(runner: Runner, session_id: str, url_info: dict):
    current_url = url_info["url"]
    print(f"\\nProcessing Competitor URL: {current_url}")

    parsed_data = await _run_agent_and_parse(runner, session_id, current_url)

    if parsed_data:
        parsed_data[URL_COL_ANALYSIS_SHEET] = current_url
        write_analysis_data_csv(parsed_data)
        mark_url_as_analyzed_csv(url_info)
        print(f"Successfully processed competitor URL and saved CSV data for {current_url}.")
    else:
        print(f"Skipping CSV update for competitor URL {current_url} due to processing/parsing failure.")

async def process_single_posted_url(runner: Runner, session_id: str, url_info: dict):
    current_url = url_info["url"]
    print(f"\\nProcessing Posted URL: {current_url}")

    parsed_data = await _run_agent_and_parse(runner, session_id, current_url)

    if parsed_data:
        update_posted_csv_data(current_url, parsed_data)
        print(f"Successfully processed posted URL and updated CSV data for {current_url}.")
    else:
        print(f"Skipping CSV update for posted URL {current_url} due to processing/parsing failure.")

async def main():
    print("Starting SEO Competitor Analysis Process (CSV Mode)...")
    global COMPETITOR_URLS_CSV_PATH, ANALYSIS_OUTPUT_CSV_PATH, POSTED_CSV_PATH

    script_dir = os.path.dirname(os.path.abspath(__file__))
    COMPETITOR_URLS_CSV_PATH = os.path.join(script_dir, f"{COMPETITOR_URLS_SHEET_NAME}.csv")
    ANALYSIS_OUTPUT_CSV_PATH = os.path.join(script_dir, f"{ANALYSIS_OUTPUT_SHEET_NAME}.csv")
    POSTED_CSV_PATH = os.path.join(script_dir, POSTED_CSV_FILENAME)

    print(f"Script execution directory: {os.path.abspath(script_dir)}")
    print(f"Using Competitor URLs CSV: {os.path.abspath(COMPETITOR_URLS_CSV_PATH)}")
    print(f"Using Analysis Output CSV: {os.path.abspath(ANALYSIS_OUTPUT_CSV_PATH)}")
    print(f"Using Posted URLs CSV: {os.path.abspath(POSTED_CSV_PATH)}")

    initialize_csv_files()

    competitor_urls_to_process = get_urls_to_analyze_csv()
    if not competitor_urls_to_process:
        print(f"No URLs found to process in '{COMPETITOR_URLS_CSV_PATH}'.")
    else:
        print(f"Found {len(competitor_urls_to_process)} Competitor URLs to analyze from CSV.")

    posted_urls_to_process = get_posted_urls_to_analyze()
    if not posted_urls_to_process:
        print(f"No URLs found to process in '{POSTED_CSV_PATH}'.")
    else:
        print(f"Found {len(posted_urls_to_process)} Posted URLs to analyze from CSV.")

    if not competitor_urls_to_process and not posted_urls_to_process:
        print("No URLs to process from any source. Exiting.")
        return

    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    session = session_service.create_session(user_id='analyzer_user', app_name='seo_analyzer_app')
    print(f"Created ADK session: {session.id} for all URL processing.")

    runner = Runner(
        app_name='seo_analyzer_app',
        agent=analyzer_agent,
        session_service=session_service,
        artifact_service=artifact_service
    )

    if competitor_urls_to_process:
        print("\\n--- Processing Competitor URLs ---")
        for url_info in competitor_urls_to_process:
            await process_single_competitor_url(runner, session.id, url_info)
            # await asyncio.sleep(1)

    if posted_urls_to_process:
        print("\\n--- Processing Posted URLs ---")
        for url_info in posted_urls_to_process:
            await process_single_posted_url(runner, session.id, url_info)
            # await asyncio.sleep(1)

    print("\\nCompetitor analysis process (CSV Mode) finished for all specified files.")

if __name__ == "__main__":
    asyncio.run(main()) 