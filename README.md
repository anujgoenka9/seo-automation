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

The project implements a comprehensive SEO content workflow, broken down into distinct, automated stages:

*   **Modular Scripting for SEO Pipeline**: The automation is divided into three core Python scripts (`analyzer.py`, `keyword_planner.py`, `blog_post_generator.py`), each representing a key phase in the SEO content creation lifecycle. This modularity allows for targeted execution and easier maintenance.
    *   `analyzer.py`: Focuses on understanding the competitive landscape and your existing content. It fetches and processes competitor URLs from `Competitor URLs.csv` and your own published content from `Posted.csv`. Using an LLM, it extracts key SEO elements (Topic, Keywords, Summary) and stores this structured data in `Competitor Analysis.csv` and updates `Posted.csv`.
    *   `keyword_planner.py`: Takes insights from `Competitor Analysis.csv` and user-defined client services/topics to generate strategic content plans. It employs a two-step LLM prompting process to first brainstorm pillar post ideas and then develop detailed keyword clusters around them. These are saved in `Clusters.csv`.
    *   `blog_post_generator.py`: Automates the creation of long-form blog content. It selects an unprocessed keyword cluster from `Clusters.csv`, then orchestrates a multi-step LLM interaction for:
        1.  Generating a preliminary content plan.
        2.  Performing online research using Perplexity models (via LiteLLM) to gather current information and sources, including extracting citation URLs.
        3.  Developing a detailed, research-backed content plan.
        4.  Writing the full blog post.
        5.  Integrating internal links by referencing analyzed content in `Posted.csv`.
        6.  Converting the final Markdown content into a styled HTML file, saved in the `generated_blog_posts/` directory.

*   **Advanced LLM Integration via ADK and LiteLLM**:
    *   **Google's Agent Development Kit (ADK)**: Provides a robust framework for building and running AI agents. The scripts utilize ADK for managing agent state, tool usage (though currently minimized in `analyzer.py`), and asynchronous communication with LLMs.
    *   **LiteLLM with OpenRouter**: Offers a unified interface to a wide array of LLMs (OpenAI, Google Gemini, Anthropic Claude, Perplexity Llama, etc.) through OpenRouter. This allows for flexibility in choosing the best model for each specific task (e.g., `gpt-4o-mini-search-preview` for analysis, Perplexity models for research). API key management is handled via a `.env` file.

*   **Data-Driven Workflow with CSVs**:
    *   All input, intermediate data, and output are managed through CSV files (`Competitor URLs.csv`, `Competitor Analysis.csv`, `Posted.csv`, `Clusters.csv`).
    *   This approach provides transparency, allows for easy manual review or modification of data at any stage, and facilitates a clear handoff between different scripts in the pipeline.
    *   Scripts include logic for initializing CSV files with correct headers if they don't exist and for robustly reading/writing data.

*   **Automated Online Research & Citation**:
    *   The `blog_post_generator.py` script integrates a crucial research step. It uses Perplexity models (known for their web-searching capabilities) to gather up-to-date information relevant to the blog post topic.
    *   It is designed to extract and potentially incorporate citation URLs from the research, enhancing the credibility and SEO value of the generated content.

*   **Strategic Internal Linking**:
    *   To improve site structure and SEO, `blog_post_generator.py` automatically identifies opportunities for internal links.
    *   It references the `Posted.csv` file (which contains URLs and summaries of your already analyzed and published content) to find relevant posts to link to within the newly generated blog post.

*   **Content to HTML Conversion**:
    *   The final output of `blog_post_generator.py` is not just raw text, but a well-structured and styled HTML file. This makes the content ready for direct publishing or easier integration into a CMS.
    *   The HTML conversion includes basic styling to ensure readability.

*   **Configuration and Extensibility**:
    *   The use of `.env` for API keys and clear CSV structures makes the system configurable.
    *   The modular design and reliance on ADK/LiteLLM allow for easier extension, such as adding new analysis tools, supporting different LLMs, or integrating new steps into the SEO workflow.

---

This README provides a general guide. You may need to adjust paths or configurations based on your specific setup or if you modify the script logic.
