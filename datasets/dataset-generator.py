import pypdfium2 as pdfium
import PIL
import os, sys, time
import pandas as pd
import logging
import pathlib
from PIL import Image
from pdf2image import convert_from_path
import pymupdf
import time
from datetime import datetime
import re
import pandas as pd
from PIL import Image
import io
import pickle


root_dir = pathlib.Path().resolve().parent
dpi_list = list(range(10,0,-1)) + list(range(300, 49, -50)) + list(range(40,10, -10))

# Konfigurasi logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helper import calculate_cmyk_percentage, log

class PDFConverter:
    df_dict = {
        'library': [],
        'dpi': [],
        'converting_time': [],
        'page': [],
        'c': [],
        'm': [],
        'y': [],
        'k': [],
        'sum': [],
    }

    
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    
    def pdf2img_converter(self, dpi) -> pd.DataFrame:
        self.reset_df_dict()
        
        pdf = convert_from_path(self.file_path, dpi=dpi, fmt='jpg')
        
        for index, page in enumerate(pdf):
            start = time.time()
            c, m, y, k = calculate_cmyk_percentage(page)
            self.df_dict_appender(c, m, y, k, index, start, dpi)
        
        return pd.DataFrame(self.df_dict)


    def pymupdf_converter(self, dpi) -> pd.DataFrame:
        self.reset_df_dict()
        pdf = pymupdf.open(self.file_path)
       
        
        for index, page in enumerate(pdf):
            start = time.time()
            pixmap = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pixmap.tobytes()))
    
            c, m, y, k = calculate_cmyk_percentage(img)
            self.df_dict_appender(c, m, y, k, index, start, dpi)
    
        return pd.DataFrame(self.df_dict)

    def pdfium_converter(self, dpi) -> pd.DataFrame:
        self.reset_df_dict()
        start = time.time()
        pdf = pdfium.PdfDocument(self.file_path)
        
        for index, page in enumerate(pdf):
            start = time.time()
            bitmap = pdf[index].render(
                scale = 1/72 * dpi,
            )
            img = bitmap.to_pil()
            c, m, y, k = calculate_cmyk_percentage(img)
            self.df_dict_appender(c, m, y, k, index, start, dpi)
    
        return pd.DataFrame(self.df_dict)

    def reset_df_dict(self):
         self.df_dict = {
            'library': [],
             'dpi': [],
            'converting_time': [],
            'page': [],
            'c': [],
            'm': [],
            'y': [],
            'k': [],
            'sum': [],
         }

    def df_dict_appender(self, c, m, y, k, index, start, dpi):
        sum_ = c + m + y + k
        self.df_dict['library'].append('pdfium')
        self.df_dict['dpi'].append(dpi)
        self.df_dict['page'].append(index+1)
        self.df_dict['c'].append(c)
        self.df_dict['m'].append(m)
        self.df_dict['y'].append(y)
        self.df_dict['k'].append(k)
        self.df_dict['sum'].append(sum_)
        self.df_dict['converting_time'].append(time.time() - start)



def generate_dataset():
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'statistik-indonesia-2024-combined.pdf',)
    conv = PDFConverter(pdf_path)
    df_dict = {
        'library': [],
        'dpi': [],
        'converting_time': [],
    }


    for dpi in dpi_list:
        logging.info(f"Converting {dpi} dpi")
        start = time.time()

        df_temp = conv.pdf2img_converter(dpi)
        df_dict['library'].append('pdf2mg')
        df_dict['dpi'].append(dpi)
        df_dict['converting_time'].append(time.time() - start)

        df_temp.to_csv(root_dir.joinpath("outputs/csv/cmyk_of_a_pdf_file_by_dpi.csv"), mode='a', index=False, header=False)

        start = time.time()
        df_temp = conv.pymupdf_converter(dpi)
        df_dict['library'].append('pymupdf')
        df_dict['dpi'].append(dpi)
        df_dict['converting_time'].append(time.time() - start)

        df_temp.to_csv(root_dir.joinpath("outputs/csv/cmyk_of_a_pdf_file_by_dpi.csv"), mode='a', index=False, header=False)

        start = time.time()
        df_temp = conv.pdfium_converter(dpi)
        df_dict['library'].append('pdfium')
        df_dict['dpi'].append(dpi)
        df_dict['converting_time'].append(time.time() - start)

        df_temp.to_csv(root_dir.joinpath("outputs/csv/cmyk_of_a_pdf_file_by_dpi.csv"), mode='a', index=False, header=False)


    df_cvt = pd.DataFrame(df_dict)
    df_cvt.to_csv(root_dir.joinpath("outputs/csv/pdf_to_img_converting_time_by_libraries.csv"), index=False)


if __name__ == '__main__':
    generate_dataset()