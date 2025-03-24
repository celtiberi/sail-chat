from typing import Optional, Dict, Any
from langchain.tools import tool
from pydantic import BaseModel, Field
import httpx
import json
from PIL import Image
import io

class BoundingBox(BaseModel):
    """Input for the WindDataTool."""
    min_lat: float = Field(
        default=37.5,
        description="Minimum latitude (-90 to 90)",
        ge=-90.0,
        le=90.0
    )
    max_lat: float = Field(
        default=42.5,
        description="Maximum latitude (-90 to 90)",
        ge=-90.0,
        le=90.0
    )
    min_lon: float = Field(
        default=-72.5,
        description="Minimum longitude (-180 to 180)",
        ge=-180.0,
        le=180.0
    )
    max_lon: float = Field(
        default=-67.5,
        description="Maximum longitude (-180 to 180)",
        ge=-180.0,
        le=180.0
    )

async def wind_data_tool(
    bounding_box: BoundingBox,
    endpoint: str = "http://localhost:8000"
) -> Dict[str, Any]:
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint}/wind-data",
                json={
                    "min_lat": bounding_box["min_lat"],
                    "max_lat": bounding_box["max_lat"],
                    "min_lon": bounding_box["min_lon"],
                    "max_lon": bounding_box["max_lon"]
                }
            )
            response.raise_for_status()
            return response.json()
                
    except Exception as e:
        return {"error": f"Failed to fetch wind data: {str(e)}"} 
    
    