from functools import lru_cache
from pydantic import BaseSettings, Field
from typing import List

class Settings(BaseSettings):
    hf_token: str = Field('', env='HF_TOKEN')
    sam2_endpoint_url: str = Field('https://api-inference.huggingface.co/models/facebook/sam2-hiera-large', env='SAM2_ENDPOINT_URL')
    openai_api_key: str = Field('', env='OPENAI_API_KEY')
    min_iou: float = Field(0.5, env='MIN_IOU')
    min_stability: float = Field(0.7, env='MIN_STABILITY')
    min_area_px: int = Field(500, env='MIN_AREA_PX')

    class Config:
        env_file = '.env'

ARCH_LABELS: List[str] = [
    'door', 'balcony', 'window', 'roof', 'wall', 'garage', 'gate', 'chimney',
    'porch', 'stairs', 'railing', 'column', 'arch', 'awning'
]

DOOR_HARDWARE = ['knob', 'lever', 'pull', 'pushbar', 'unknown']
BALCONY_STRUCTURES = ['recessed', 'protruding', 'cantilevered', 'decorative']
BALCONY_VIEW = ['single_direction', 'multi_direction']
BALCONY_DOOR = ['single_door', 'double_doors', 'french_doors', 'sliding_doors']
BALCONY_WINDOWS = ['single', 'double', 'none']
BALCONY_RAILING = ['half_wall', 'plain', 'ornamental']


def enum_guard(value: str, allowed: List[str], default: str) -> str:
    return value if value in allowed else default


@lru_cache()
def get_settings() -> Settings:
    return Settings()
