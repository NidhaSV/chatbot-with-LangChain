from pydantic import BaseModel

class QueryRequest(BaseModel):
    """Request model for the /llm/query endpoint."""
    prompt: str

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
