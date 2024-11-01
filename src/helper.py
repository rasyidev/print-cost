import shutil
import os
import re
import PIL
from PIL import Image
import numpy as np
import pypdfium2 as pdfium
import pickle
import time
import pandas as pd
from werkzeug.datastructures.file_storage import FileStorage
import random
import matplotlib.pyplot as plt
import json

def log(file_path: str, text: str) -> None:
    """
    Send log into log file using append method
    file_path: str
    text: str
    """
    with open(file_path, 'a') as f:
        f.writelines(f"\n{text}")

def label_to_price(df_output, verbose=False):
    """df_output: output of predict_pdf function"""
    df = df_output.copy()
    price_label = json.load(open('../models/price_label.json'))
    price_map = {}
    for i, price in enumerate(price_label['prices']):
        price_map[i] = price  
    df['price'] = df['label'].map(price_map)
    n_pages_500 = len(df[df['price'] == 500])
    n_pages_1000 = len(df[df['price'] == 1000])
    n_pages_1500 = len(df[df['price'] == 1500])
    n_pages_2000 = len(df[df['price'] == 2000])
    print(df)

    if verbose:
        print(f"""
            500  x n_pages_500 \t= {500 * n_pages_500}
            1000 x n_pages_1000 \t= {1000 * n_pages_1000}
            2000 x n_pages_2000 \t= {2000 * n_pages_2000}
        """)
    output = {
                'items': [
                            (n_pages_500, 500),
                            (n_pages_1000, 1000),
                            (n_pages_1500, 1500),
                            (n_pages_2000, 2000),
                        ],
                'price_total': sum(df['price'])
           }
    del df
    return output
 

def predict_pdf(file_path_or_pdf_bytes):
    # Check file type, must be pdf
    if not type(file_path_or_pdf_bytes) == str and isinstance(file_path_or_pdf_bytes, FileStorage) == False:
        raise Exception("FileTypeERROR: File must be a path or pdf file bytes")
    
    if type(file_path_or_pdf_bytes) == str:
        if file_path_or_pdf_bytes.split('.')[-1].lower() != 'pdf':
            raise Exception("FileExtensionError: File must be pdf")
        
    # Check file page type, must be A4
    # later
    output = {'time': [], 'label': []}
    
    pdf = pdfium.PdfDocument(file_path_or_pdf_bytes)
    for i in range(len(pdf)):
        start = time.time()
        bitmap = pdf[i].render(
            scale = 1/72 * 5, # 5 DPI
            rotation = 0, 
        )
        pil_image = bitmap.to_pil()
        res = sum(calculate_cmyk_percentage(pil_image))
        kmeans, scaler = pickle.load(open('../models/kmeans_and_scaler.pkl', 'rb'))
        label_pred = kmeans.predict(scaler.transform([pd.Series({'sum': res})]))[0]
        print(f"page-{i+1}:", label_pred)
        
        output['time'].append(time.time() - start)
        output['label'].append(label_pred)
        del start, bitmap, pil_image, res, kmeans, scaler, label_pred
    return pd.DataFrame(output)
        
        

def get_folder_size(folder_path: str) -> float:
    """
    Get size of specific folder
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return round(total_size / (1024 * 1024), 2)  # Mengonversi byte ke MB


def rmtree(folder_path:str) -> None:
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' telah dihapus beserta isinya.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' tidak ditemukan.")
    except PermissionError:
        print(f"Tidak memiliki izin untuk menghapus folder '{folder_path}'.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

def prepend_zero(file_name):
    """
    Prepend string contains number with 0 up to 999
    example: something-1.jpg -> something-001.jpg
    """
    number = int(re.findall("\d+", file_name)[0])
    result = 'page-'
    if number / 10 < 1:
        result = f"{result}00{number}.jpg"
    elif number / 10 < 10:
        result = f"{result}0{number}.jpg"
    else:
        result = f"{result}{number}.jpg"

    return result

# Fungsi untuk konversi RGB ke CMYK
def rgb_to_cmyk(image_path_or_pil_img):
    # print(type(image_path_or_pil_img) == str, isinstance(image_path_or_pil_img, PIL.Image.Image))
    if not type(image_path_or_pil_img) == str and not isinstance(image_path_or_pil_img, PIL.Image.Image):
        raise Exception("ImportError: image_path_or_pil_img must be str or PIL.Image.Image, not", type(image_path_or_pil_img))

    if type(image_path_or_pil_img) == str:
        if image_path_or_pil_img.split('.')[-1].lower() not in ['jpg', 'png', 'jpeg', 'webp', 'heic']:
            raise Exception("FileExtensionError: file type must be an image format")
    
    # Membuka gambar dari path dan konversi ke RGB
    img = None
    if type(image_path_or_pil_img) == str:
        img = Image.open(image_path_or_pil_img).convert("RGB")

    else:
        img = image_path_or_pil_img

    # Konversi gambar menjadi array NumPy
    img_array = np.array(img) / 255.0
    
    # Konversi RGB ke CMY
    c = 1 - img_array[:, :, 0]
    m = 1 - img_array[:, :, 1]
    y = 1 - img_array[:, :, 2]
    
    # Hitung nilai K (Black)
    k = np.minimum(np.minimum(c, m), y)
    
    # Menghindari pembagian oleh nol
    c = (c - k) / (1 - k + 1e-10)
    m = (m - k) / (1 - k + 1e-10)
    y = (y - k) / (1 - k + 1e-10)
    
    # Gabungkan dalam array CMYK
    return np.dstack((c, m, y, k))


# Fungsi untuk menghitung persentase komponen CMYK
def calculate_cmyk_percentage(image_path):
    # Konversi RGB ke CMYK
    cmyk_array = rgb_to_cmyk(image_path)
    
    # Menghitung rata-rata persentase dari setiap channel (C, M, Y, K)
    c_percent = np.mean(cmyk_array[:, :, 0]) * 100
    m_percent = np.mean(cmyk_array[:, :, 1]) * 100
    y_percent = np.mean(cmyk_array[:, :, 2]) * 100
    k_percent = np.mean(cmyk_array[:, :, 3]) * 100
    
    # # Output persentase
    # print(f"Persentase C: {c_percent:.2f}%")
    # print(f"Persentase M: {m_percent:.2f}%")
    # print(f"Persentase Y: {y_percent:.2f}%")
    # print(f"Persentase K: {k_percent:.2f}%")

    return round(c_percent, 2), round(m_percent, 2), round(y_percent, 2), round(k_percent, 2)

# Fungsi menampilkan 8 gambar untuk label tertentu
def show_clustered_image(df, label):
    img_index_tobe_showed = list(df[df['label']==label].index)

    # Show 8 image only
    if len(img_index_tobe_showed) > 8:
        img_index_tobe_showed = random.sample(img_index_tobe_showed, 8)

    fig, ax = plt.subplots(1, len(img_index_tobe_showed))
    
    for i, img_index in enumerate(img_index_tobe_showed):
        plt.grid = False
        ax[i].imshow(plt.imread(f'../outputs/pdfium_200dpi/{prepend_zero(str(img_index+1))}'))
        ax[i].set_title(f"page-{img_index+1}")

    fig.set_figheight(10)
    fig.set_figwidth(60)