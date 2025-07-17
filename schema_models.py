from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, validator

from config import (
    DOOR_HARDWARE,
    BALCONY_STRUCTURES,
    BALCONY_VIEW,
    BALCONY_DOOR,
    BALCONY_WINDOWS,
    BALCONY_RAILING,
    enum_guard,
)

class DoorDetail(BaseModel):
    leafCount: int
    isGlazed: bool
    hardware: str

    @validator('hardware', pre=True)
    def valid_hardware(cls, v):
        return enum_guard(str(v), DOOR_HARDWARE, 'unknown')

class BalconyDetail(BaseModel):
    structure: str
    view: str
    door: str
    windows: str
    railing: str

    @validator('structure', pre=True)
    def val_structure(cls, v):
        return enum_guard(str(v), BALCONY_STRUCTURES, 'decorative')

    @validator('view', pre=True)
    def val_view(cls, v):
        return enum_guard(str(v), BALCONY_VIEW, 'single_direction')

    @validator('door', pre=True)
    def val_door(cls, v):
        return enum_guard(str(v), BALCONY_DOOR, 'single_door')

    @validator('windows', pre=True)
    def val_windows(cls, v):
        return enum_guard(str(v), BALCONY_WINDOWS, 'none')

    @validator('railing', pre=True)
    def val_railing(cls, v):
        return enum_guard(str(v), BALCONY_RAILING, 'plain')

class TagItem(BaseModel):
    name: str
    category: str
    confidence: float

class ColorItem(BaseModel):
    color: str
    hex: str
    objects: List[str]

class CameraData(BaseModel):
    position: str
    isProfessionalPhoto: bool
    cameraType: str
    technique: str
    framing: str
    timeOfDay: str
    lighting: str

class FileData(BaseModel):
    fileSizeBytes: int
    width: int
    height: int
    dimensions: str
    aspectRatio: float
    horizontalResolutionDPI: int
    verticalResolutionDPI: int
    bitDepth: int
    colorMode: str
    format: str
    imageQuality: str

class ImageSchema(BaseModel):
    imageType: str
    doorDetails: List[DoorDetail] = Field(default_factory=list)
    balconyDetails: List[BalconyDetail] = Field(default_factory=list)
    tags: List[TagItem] = Field(default_factory=list)
    colors: List[ColorItem] = Field(default_factory=list)
    cameraData: Optional[CameraData] = None
    fileData: Optional[FileData] = None
    roofDescription: Optional[str] = ''
    description: Optional[str] = ''
    contentFlagged: bool = False

    def to_json_dict(self) -> dict:
        return self.dict(by_alias=True, exclude_none=True)

    @staticmethod
    def empty_schema() -> 'ImageSchema':
        return ImageSchema(imageType='Exterior')

    @staticmethod
    def from_partial(data: dict) -> 'ImageSchema':
        return ImageSchema.parse_obj(data)
