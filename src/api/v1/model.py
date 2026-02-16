"""Model information endpoint.

This module provides an endpoint to get information about the deployed model.
"""

from fastapi import APIRouter, Depends, status
from loguru import logger

from src.api.dependencies import verify_api_key
from src.core.app_state import get_model_service
from src.schemas.prediction import ModelInfoResponse
from src.utils.response import generate_response

router = APIRouter()


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Retrieve information about the currently deployed model",
)
async def get_model_info(api_key: str = Depends(verify_api_key)):
    """Get information about the currently loaded model.

    Args:
        api_key: Validated API key

    Returns:
        ModelInfoResponse with model metadata

    Raises:
        HTTPException: If model info cannot be retrieved
    """
    try:
        model_service = get_model_service()
        model_info = model_service.get_model_info()

        logger.info(f"Model info requested: {model_info['model_name']} v{model_info['model_version']}")

        return generate_response(
            success=True,
            data=model_info,
            message="Model information retrieved successfully",
            status_code=200
        )

    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        return generate_response(
            success=False,
            error="model_info_error",
            exception_error=str(e),
            message="Failed to retrieve model information",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
