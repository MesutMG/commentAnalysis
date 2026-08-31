# Customer Comment Analysis & Moderation Pipeline

A hybrid NLP pipeline designed to process, moderate, and analyze Turkish customer feedback. It combines static regex filtering with an asynchronous local Large Language Model (LLM) via Ollama to extract sentiment, categorize complaints, and automatically correct mismatched star ratings.

It can be run as a standalone CLI script for batch processing or deployed as a live REST API using FastAPI.

## Features

* **Hybrid Filtering:** Uses lightweight, precompiled regex rules to catch explicit profanity, spam etc. before spending LLM compute.

* **LLM Integration:** Asynchronously queries a local LLM to determine sentiment, category, and nuanced inappropriateness.

* **Pydantic Validation:** Ensures the LLM's JSON outputs match to the expected schema and handles probable hallucinations.

* **Star Correction Logic:** Adjusting star ratings if they contradict the sentiment.\
(e.g., a 5-star rating on a heavily negative comment)

* **Dual Execution Mode:** Run batch CSV processing via terminal, or serve endpoints dynamically via FastAPI.

## Project Structure

```text
.
├── config.json               # Local configuration file (ignored in git)
├── config_example.json       # Example configuration template
├── main.py                   # Entry point (CLI & FastAPI app)
├── requirements.txt          # Python dependencies
└── src
    ├── analyze.py            # Async LLM connection and Pydantic validation
    ├── filter.py             # Static regex rule engine for Turkish moderation
    └── preprocess.py         # Data cleaning (dropping empty rows, PII)

```

## Prerequisites

1. **Python 3.10+**
2. **Ollama:** Must be installed and running on the host machine or server.
3. **LLM Model:** Pull the required model before running the pipeline.\
Such as,
```bash
ollama pull qwen2.5:3b

# or

ollama pull qwen2.5:7b
# Recommended for server deployment
```



## Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate 

# Windows:
python -m venv venv
source venv\Scripts\activate
```


3. Install the dependencies:
```bash
pip install -r requirements.txt
```


4. Set up your configuration:
```bash
cp config_example.json config.json
```


*Edit `config.json` to match your local paths, preferred model, and concurrency limits.*

## Configuration (`config.json`)

```json
{
  "data_path": "data/example_comments.csv",
  "output_path": "output.csv",
  "ollama_url": "http://localhost:11434/api/generate",
  "model_name": "qwen2.5:3b",
  "concurrency_limit": 2,
  "llm_rows": null
}

```

* `concurrency_limit`: Controls how many async requests hit Ollama simultaneously. Keep this low (1-2) on local machines to prevent thermal throttling, but you can increase it on dedicated servers.
* `llm_rows`: Set to an integer to sample a specific number of rows for testing, or *`null`* to process the entire dataset.

## Usage

### 1. CLI Batch Processing

To process the CSV file defined in `config.json` and generate `output.csv` and `output_cleaned.csv`:

```bash
python3 main.py
```

### 2. FastAPI Server

To launch the live API for dynamic requests:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 1967
```

You can view the interactive API documentation (Swagger UI) by navigating to `http://localhost:1967/docs` in your browser.

## API Endpoints

### `POST /analyzeComment/`

Analyzes a single comment synchronously without batching overhead.
**Request Body:**

```json
{
  "comment": "Lavabolar berbat hijyen sıfır...",
  "star": 5.0
}
```

### `POST /analyzeWholeCsv/`

Upload a `.csv` file via multipart/form-data to process the entire dataset and receive the analyzed records as a JSON response.

---

## Testing

Run the included bash test script to verify both single-comment analysis and batch CSV upload against the running server:

```bash
chmod +x test_api.sh
./test_api.sh > output.txt