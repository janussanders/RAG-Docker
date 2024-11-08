from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from document_processor import DocumentProcessor
from loguru import logger

app = FastAPI(title="Document Processing API")

@app.on_event("startup")
async def startup_event():
    """Initialize the document processor on startup."""
    app.state.processor = DocumentProcessor()
    await app.state.processor.initialize()
    logger.info("Document processor initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    await app.state.processor.close()
    logger.info("Document processor closed")

@app.post("/index/create")
async def create_index():
    """Create index from documents."""
    try:
        index = await app.state.processor.create_index()
        return {"status": "success", "message": "Index created successfully"}
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[str])
async def get_documents():
    """Load and return documents."""
    try:
        documents = await app.state.processor.load_documents()
        return documents
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collection/info")
async def get_collection_info():
    """Get information about the current collection."""
    try:
        info = await app.state.processor.get_collection_info()
        return JSONResponse(content=info.dict())
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 