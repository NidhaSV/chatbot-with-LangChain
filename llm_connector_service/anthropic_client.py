import os
import httpx
import logging
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnthropicClient:
    """A client for interacting with the Anthropic Claude API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        self._api_key = api_key
        self._headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def stream_completion(self, prompt: str):
        """
        Sends a prompt to the Claude API and streams the response.

        Args:
            prompt: The prompt to send to the model.

        Yields:
            The content chunks from the API response.
        
        Raises:
            httpx.HTTPStatusError: If the API returns an error status code.
        """
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream("POST", ANTHROPIC_API_URL, headers=self._headers, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except httpx.HTTPStatusError as e:
                logger.error(f"API Error: {e.response.status_code} - {e.response.text}")
                if e.response.status_code == 401:
                    raise httpx.HTTPStatusError("Invalid Anthropic API key.", request=e.request, response=e.response)
                elif e.response.status_code == 429:
                    raise httpx.HTTPStatusError("Rate limit exceeded. Please try again later.", request=e.request, response=e.response)
                else:
                    raise
            except httpx.TimeoutException as e:
                logger.error(f"Request timed out: {e}")
                raise httpx.HTTPStatusError("Request to Anthropic API timed out.", request=e.request, response=httpx.Response(504))


def get_anthropic_client() -> AnthropicClient:
    """
    Dependency injector for the AnthropicClient.
    
    Returns:
        An instance of the AnthropicClient.
    """
    return AnthropicClient(api_key=ANTHROPIC_API_KEY)
