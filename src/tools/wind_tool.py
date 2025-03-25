from typing import Optional, Dict, Any, Union
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
    input_data: Union[BoundingBox, str],
    endpoint: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Get wind data for either a bounding box or location name.
    
    Args:
        input_data: Either a BoundingBox object or a location name string
        endpoint: The API endpoint URL
    """
    try:
        async with httpx.AsyncClient() as client:
            if isinstance(input_data, str):
                # If input is a string, treat it as a location name
                response = await client.post(
                    f"{endpoint}/wind-data",
                    json={"name": input_data}
                )
            else:
                # If input is a BoundingBox, use its coordinates
                response = await client.post(
                    f"{endpoint}/wind-data",
                    json={
                        "min_lat": input_data.min_lat,
                        "max_lat": input_data.max_lat,
                        "min_lon": input_data.min_lon,
                        "max_lon": input_data.max_lon
                    }
                )
            response.raise_for_status()
            return response.json()
                
    except Exception as e:
        return {"error": f"Failed to fetch wind data: {str(e)}"} 
    
    