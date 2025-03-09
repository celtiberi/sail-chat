import os
import sys
from fastapi import FastAPI
from chainlit.utils import mount_chainlit
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

app = FastAPI()

@app.get("/api")
def read_main():
    return {"message": "Sailors Parrot API"}

# Mount the Chainlit app with the correct path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, "app.py")
# leave path="/" we want to load the app at the root
mount_chainlit(app=app, target=app_path, path="/")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the FastAPI application shuts down."""
    logger.info("FastAPI application shutting down, cleaning up resources...")
    try:
        # No need to clean up the visual index as it's now handled by the separate service
        logger.info("Application resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("CHAINLIT_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 