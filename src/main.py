import os
import sys
from fastapi import FastAPI
from chainlit.utils import mount_chainlit

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

app = FastAPI()

@app.get("/app")
def read_main():
    return {"message": "Hello World from main app"}

# Mount the Chainlit app with the correct path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, "app.py")
mount_chainlit(app=app, target=app_path, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("CHAINLIT_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 