from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from httpx import HTTPStatusError

from models import QueryRequest, ErrorResponse
from anthropic_client import AnthropicClient, get_anthropic_client
from retrieval import sap_retrieval, SAPDefinition
import logging
from typing import AsyncGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/health", responses={
    200: {"content": {"application/json": {}}}
})
async def checkHealth():
    return {"status": "OK"}

@app.post("/llm/query", responses={
    200: {"content": {"text/event-stream": {}}},
    401: {"model": ErrorResponse, "description": "Invalid API Key"},
    429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"},
    504: {"model": ErrorResponse, "description": "Gateway Timeout"}
})
async def query_llm(
    request: QueryRequest,
    client: AnthropicClient = Depends(get_anthropic_client)
) -> StreamingResponse:
    """
    Accepts a prompt, retrieves relevant SAP definitions, augments the prompt, and streams responses from the Anthropic Claude API.
    """
    try:
        defs_text = "Do not guess or make up expansion for abbreviations."
        # Retrieve top-3 SAP definitions
        retrieved_defs = sap_retrieval.query(request.prompt, top_k=3)
        # Format definitions
        defs_text += "Relevant SAP definitions:\n"
        for d, _ in retrieved_defs:
            defs_text += f"- {d.field_name} ({d.data_type}): {d.description}\n"
        defs_text += f"\nUser Query: {request.prompt}"
        logger.info(f"Augmented prompt sent to LLM:\n{defs_text}")
        # Stream response from LLM
        return StreamingResponse(
            client.stream_completion(defs_text),
            media_type="text/event-stream"
        )
    except HTTPStatusError as e:
        status_code = e.response.status_code
        detail = str(e)
        if status_code == 401:
            return JSONResponse(status_code=401, content={"detail": "Invalid Anthropic API key."})
        elif status_code == 429:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please try again later."})
        elif status_code == 504:
            return JSONResponse(status_code=504, content={"detail": "Request to Anthropic API timed out."})
        else:
            return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred with the Anthropic API."})
    except Exception as e:
        logger.error(f"Unexpected server error: {e}")
        return JSONResponse(status_code=500, content={"detail": f"An unexpected server error occurred: {e}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
