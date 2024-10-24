import shutil
import os
import re
from PIL import Image
import numpy as np

def log(file_path: str, text: str) -> None:
    """
    Send log into log file using append method
    file_path: str
    text: str
    """
    with open(file_path, 'a') as f:
        f.writelines(f"\n{text}")


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
def rgb_to_cmyk(image_path):
    # Membuka gambar dari path dan konversi ke RGB
    img = Image.open(image_path).convert("RGB")
    
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
    
    # Output persentase
    print(f"Persentase C: {c_percent:.2f}%")
    print(f"Persentase M: {m_percent:.2f}%")
    print(f"Persentase Y: {y_percent:.2f}%")
    print(f"Persentase K: {k_percent:.2f}%")