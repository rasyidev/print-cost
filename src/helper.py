def log(file_path: str, text: str) -> None:
    """
    Send log into log file using append method
    file_path: str
    text: str
    """
    pass


import os

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


