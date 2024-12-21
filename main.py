from fastapi import FastAPI, UploadFile
from src.helper import PrintCost

import pandas as pd
import pymupdf

import time


app = FastAPI()


@app.post("/print-cost/")
async def calculate_print_cost(file: UploadFile):
    pdf = pymupdf.open(stream=file.file.read())
    pc = PrintCost(pdf, "models/xgboost_98.64_cmy_k_cmyk_7_dpi.pkl")
    result = pc.predict(dpi=7)

    return {"result": result}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}