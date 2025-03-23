import asyncio
import json
import os
from typing import AsyncIterator
from contextlib import asynccontextmanager

import requests
from mcp.server.fastmcp import FastMCP, Context

# Windy API configuration
WINDY_API_KEY = os.getenv("WINDY_API_KEY")  # Set this in your environment or .env file
WINDY_URL = "https://api.windy.com/api/point-forecast/v2"

if not WINDY_API_KEY:
    raise ValueError("Please set WINDY_API_KEY in your environment variables.")

# Initialize the MCP server
mcp = FastMCP("WeatherServer", dependencies=["requests"])

# Lifespan management for startup/shutdown (optional, but good practice)
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Manage server lifecycle."""
    print("Starting Weather MCP Server...")
    try:
        yield {"status": "running"}
    finally:
        print("Shutting down Weather MCP Server...")

mcp.lifespan = app_lifespan

# Helper function to fetch Windy API data
def fetch_windy_forecast(lat: float, lon: float, parameters: list[str]) -> dict:
    """Fetch weather data from Windy API synchronously."""
    body = {
        "lat": round(lat, 3),
        "lon": round(lon, 3),
        "model": "gfs",
        "parameters": parameters,
        "levels": ["surface"],
        "key": WINDY_API_KEY
    }
    response = requests.post(WINDY_URL, headers={"Content-Type": "application/json"}, data=json.dumps(body))
    if response.status_code != 200:
        raise Exception(f"Windy API error {response.status_code}: {response.text}")
    return response.json()

# MCP Resource: Expose current weather summary
@mcp.resource("weather://{lat}/{lon}")
async def get_weather_summary(lat: str, lon: str) -> str:
    """Return a weather summary for the given coordinates."""
    lat, lon = float(lat), float(lon)
    data = fetch_windy_forecast(lat, lon, ["temp", "wind", "pressure"])
    
    # Extract latest data (first timestamp)
    temp = data["temp-surface"][0] - 273.15  # Convert Kelvin to Celsius
    wind = data["wind-surface"][0]
    pressure = data["pressure-surface"][0]
    
    summary = f"Weather at ({lat}, {lon}): Temp: {temp:.1f}°C, Wind: {wind:.1f} m/s, Pressure: {pressure:.1f} hPa"
    return summary

# MCP Tool: Fetch detailed weather forecast
@mcp.tool()
def get_weather_forecast(ctx: Context, lat: float, lon: float) -> str:
    """Fetch and return detailed weather forecast as a JSON string."""
    parameters = ["temp", "wind", "pressure"]
    data = fetch_windy_forecast(lat, lon, parameters)
    
    # Simplify the response for the LLM
    forecast = {
        "timestamps": [ts / 1000 for ts in data["ts"]],  # Convert ms to seconds
        "temperature_C": [t - 273.15 for t in data["temp-surface"]],  # Kelvin to Celsius
        "wind_speed_ms": data["wind-surface"],
        "pressure_hPa": data["pressure-surface"]
    }
    
    # Log progress (optional, for long-running tasks)
    ctx.info(f"Fetched weather data for ({lat}, {lon})")
    return json.dumps(forecast, indent=2)

# Run the server
async def main():
    print("Starting MCP Weather Server...")
    await mcp.run()

if __name__ == "__main__":
    asyncio.run(main())