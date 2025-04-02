from typing import Optional, Dict, Any, Union
from langchain.tools import tool
import httpx
import json
from PIL import Image
import io
from src.schemas import BoundingBox

async def wave_data_tool(
    input_data: Union[BoundingBox, str],
    endpoint: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Get wave data for either a bounding box or location name.
    
    Args:
        input_data: Either a BoundingBox object or a location name string
        endpoint: The API endpoint URL
    """
    try:
        async with httpx.AsyncClient() as client:
            if isinstance(input_data, str):
                # If input is a string, treat it as a location name
                response = await client.post(
                    f"{endpoint}/wave-data",
                    json={"name": input_data}
                )
            else:
                # If input is a BoundingBox, use its coordinates
                response = await client.post(
                    f"{endpoint}/wave-data",
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
        return {"error": f"Failed to fetch wave data: {str(e)}"} 