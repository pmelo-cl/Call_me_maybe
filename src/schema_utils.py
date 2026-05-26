"""Utilidades para cargar definiciones de funciones."""
import json
from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel, Field


class ParameterDef(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef] = Field(default_factory=dict)
    returns: Dict[str, str]


def load_function_definitions(path: Path) -> List[FunctionDefinition]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [FunctionDefinition(**item) for item in data]
