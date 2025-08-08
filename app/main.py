from fastapi import FastAPI
from pydantic import BaseModel
from app.clean_abap_code import clean_abap_code
import os
import re

app = FastAPI()

class CodeInput(BaseModel):
    code: str

@app.post("/clean_abap/")
async def clean_abap(input_data: CodeInput):
    cleaned_code = clean_abap_code(input_data.code)
    return {"cleaned_code": cleaned_code}
