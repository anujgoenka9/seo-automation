# SEO Automation Project

## Description

This project is a suite of Python scripts designed to automate various SEO tasks, including competitor analysis, keyword planning, and blog post generation. It leverages Google's Agent Development Kit (ADK) and LiteLLM (via OpenRouter) to interact with various Large Language Models (LLMs) for content generation and analysis.

The project is primarily composed of three scripts:
1.  `analyzer.py`: Analyzes competitor URLs and your own posted blog content.
2.  `keyword_planner.py`: Generates pillar post ideas and keyword clusters.
3.  `blog_post_generator.py`: Creates full blog posts from keyword clusters, including research and HTML conversion.

## Prerequisites

*   Python >=3.12 (as specified in `pyproject.toml`)
*   `uv` Python packager and virtual environment manager.

## Setup Instructions

1.  **Ensure you are in the Project Root Directory:**
    Your terminal should be at the root of the project (e.g., the `seo_automation` directory if that's your repository name).

2.  **Create and Activate Virtual Environment using `uv`:**
    It's highly recommended to use a virtual environment.
    ```bash
    uv venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies using `uv`:**
    The project dependencies are listed in `pyproject.toml`. Install them using:
    ```bash
    uv pip install -e .
    ```
    This command installs the project in editable mode along with its dependencies.

4.  **Set Up Environment Variables:**
    The scripts require an API key for OpenRouter to access various LLMs.
    *   Create a file named `.env` in the project root (alongside `pyproject.toml` and your scripts).
    *   Add your OpenRouter API key to the `.env` file:
        ```env
        OPENROUTER_API_KEY="your_openrouter_api_key_here"
        ```
    *   The scripts are configured to load the `.env` file from their current working directory.

## Project Structure

```
.
├── analyzer.py             # Script for competitor and own content analysis
├── keyword_planner.py      # Script for keyword clustering and pillar post ideas
├── blog_post_generator.py  # Script for generating full blog posts
├── Competitor URLs.csv     # Input: List of competitor URLs to analyze
├── Competitor Analysis.csv # Output from analyzer.py / Input for keyword_planner.py
├── Posted.csv              # Input/Output: List of your posted blog URLs, topics, keywords, summaries
├── Clusters.csv            # Input/Output: Generated keyword clusters, pillar posts, and status
├── generated_blog_posts/   # Output: Directory for final HTML blog posts
├── .env                    # (You need to create this) Stores API keys
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # This file
```

## Running the Scripts

Ensure your virtual environment is activated and the `.env` file (located in the project root) is correctly set up. All scripts should be run from the project root directory.

**1. Analyzer (`analyzer.py`)**
   *   **Purpose**: Reads URLs from `Competitor URLs.csv`, analyzes them using an LLM to extract Topic, Keywords, and Summary, and writes the results to `Competitor Analysis.csv`. It also processes `Posted.csv` to analyze your own blog posts and updates `Posted.csv` in place with the analysis and marks them as "Analysed: Yes".
   *   **Input**:
        *   `Competitor URLs.csv`: Must contain a header row with at least a `URL` column.
        *   `Posted.csv`: Must contain a header row with at least a `URL` column. Other columns like `Topic`, `Keywords`, `Summary`, `Analysed` will be populated.
   *   **Output**:
        *   `Competitor Analysis.csv`: Contains analysis of competitor URLs.
        *   `Posted.csv`: Updated with analysis for your own blog URLs.
   *   **To Run**:
        ```bash
        python analyzer.py
        ```

**2. Keyword Planner (`keyword_planner.py`)**
   *   **Purpose**: Generates pillar post ideas and keyword clusters based on client services/topic, using a two-prompt process. It also incorporates keywords from `Competitor Analysis.csv`.
   *   **Input**:
        *   `Competitor Analysis.csv` (specifically the `Keywords` column).
        *   User input for the client's services/topic when prompted by the script (or modify script for direct input).
   *   **Output**:
        *   `Clusters.csv`: Contains generated keyword clusters with columns like `Cluster`, `Intent`, `Keywords`, `Primary Keyword`, and `Completed` (initially "No").
   *   **To Run**:
        ```bash
        python keyword_planner.py
        ```

**3. Blog Post Generator (`blog_post_generator.py`)**
   *   **Purpose**: Takes an unprocessed keyword cluster from `Clusters.csv`, generates a preliminary plan, conducts research (fetching and processing online sources), creates a detailed plan, writes the blog post, adds internal links (from `Posted.csv`), and finally converts the post to an HTML file.
   *   **Input**:
        *   `Clusters.csv`: Reads the next cluster marked "No" in the `Completed` column.
        *   `Posted.csv`: Used to source internal linking opportunities (URLs from posts marked "Analysed: Yes").
   *   **Output**:
        *   `generated_blog_posts/PRIMARY_KEYWORD.html`: The final HTML blog post.
        *   `Clusters.csv`: Updates the `Completed` status to "Yes" for the processed cluster.
   *   **To Run**:
        ```bash
        python blog_post_generator.py
        ```

## Key Workflow & Features

*   **Modular Design**: Each script handles a distinct phase of the SEO content pipeline.
*   **LLM Integration**: Uses ADK and LiteLLM with OpenRouter to flexibly connect to various LLMs (e.g., OpenAI GPT models, Google Gemini, Anthropic Claude, Perplexity Llama).
*   **CSV Data Management**: Input and output data are managed through CSV files, allowing for easy review and modification.
*   **Automated Research**: `blog_post_generator.py` includes a step for direct online research using Perplexity models, with citation URL extraction.
*   **Dynamic Path Handling**: Scripts are configured to locate necessary CSV files and output directories relative to their execution path (the project root).
*   **Internal Linking**: Automatically suggests and incorporates internal links into generated blog posts based on previously analyzed content in `Posted.csv`.
*   **HTML Conversion**: Produces a styled HTML output for generated blog posts.

## Dependencies

All Python dependencies are listed in the `pyproject.toml` file (located in the project root) and can be installed using `uv pip install -e .`. Key dependencies include:

*   `google-adk`: For interacting with LLMs using an agent-based framework.
*   `litellm`: For simplified interaction with a wide range of LLM APIs, primarily via OpenRouter in this project.
*   `python-dotenv`: For managing environment variables (API keys).
*   `beautifulsoup4`, `requests`, `openpyxl`: Utility libraries. (Note: `openpyxl` usage was reduced in favor of CSVs).

Refer to `pyproject.toml` for the full list and specific versions.

## Configuration

*   **API Keys**: The primary configuration is the `OPENROUTER_API_KEY` which must be set in a `.env` file located within the project root.
*   **LLM Models**: Model names for different agents are defined within each script (e.g., `PRELIM_PLAN_MODEL_NAME` in `blog_post_generator.py`). These can be changed to use different models available via OpenRouter.
*   **CSV File Paths**: Paths are generally configured at the beginning of each script and are designed to be relative to the project root (where the scripts are run).

---

This README provides a general guide. You may need to adjust paths or configurations based on your specific setup or if you modify the script logic.
