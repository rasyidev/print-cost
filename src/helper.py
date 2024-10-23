import shutil
import os

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