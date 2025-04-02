from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Input for geographical bounding box queries."""
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