import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from src.schemas.churn_models import ErrorResponse
from src.api.endpoints import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(exc.detail)
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(code=exc.status_code, message=exc.detail, details="").model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(str(exc.errors()))
    return JSONResponse(status_code=422, content=ErrorResponse(code=422, message="Неверные данные запроса", details=str(exc.errors())).model_dump())