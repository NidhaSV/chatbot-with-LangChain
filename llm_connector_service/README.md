# LLM Connector Service

This is a Python microservice built with FastAPI that acts as a connector to the Anthropic Claude API. It exposes a single endpoint to accept a prompt and streams the response back to the client.

## Features

- **FastAPI Backend**: High-performance async framework.
- **Anthropic Claude Integration**: Streams responses from the Claude v1 Messages API.
- **Environment-Based Configuration**: Uses a `.env` file for secure API key management.
- **Error Handling**: Gracefully handles common API errors (e.g., invalid key, rate limits, timeouts).
- **Async Communication**: Uses `httpx` for non-blocking requests to the Claude API.

---

## 1. Setup and Installation

### Prerequisites

- Python 3.8+
- `pip` for package management

### Installation

1.  **Clone the repository** (or download the files).

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Configuration

### Set API Key

The service requires an API key from Anthropic to authenticate requests.

1.  Create a file named `.env` in the root of the `llm_connector_service` directory.
2.  Open the file and add a line as follows (replace `<your_anthropic_api_key>` with your actual key):

    ```.env
    ANTHROPIC_API_KEY=<your_anthropic_api_key>
    ```

---

## 3. Running the Service

Once the dependencies are installed and the API key is configured, you can start the service by executing following commands in the terminal.

```bash
cd llm_connector_service
python main.py
```

The server will start and be accessible at `http://localhost:8000`.

---

## 4. Testing the Endpoint

You can test the `/llm/query` endpoint using the interactive API documentation or a tool like `curl`.

### Using the Interactive Docs

1.  With the server running, open your browser and navigate to:
    [http://localhost:8000/docs](http://localhost:8000/docs)

2.  Expand the `POST /llm/query` endpoint, click **"Try it out"**, enter your JSON payload, and click **"Execute"**.

### Using `curl`

To test the streaming response from your terminal, use the following `curl` command. The `--no-buffer` flag is important for seeing the streamed events in real-time.

```bash
curl -X POST "http://localhost:8000/llm/query" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Tell me a joke about APIs"}' \
     --no-buffer
``` 