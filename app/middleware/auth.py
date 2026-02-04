"""API Key Authentication Middleware"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader as FastAPIKeyHeader
import secrets

from ..config import settings


# API Key header configuration
APIKeyHeader = FastAPIKeyHeader(
    name="x-api-key",
    auto_error=False,
    description="API Key for authentication"
)


async def verify_api_key(api_key: str = Security(APIKeyHeader)) -> str:
    """
    Verify the API key from request headers
    
    Args:
        api_key: API key from x-api-key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Check if API key is provided
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "API key required. Please provide x-api-key header."
            }
        )
    
    # Secure comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
    
    return api_key
