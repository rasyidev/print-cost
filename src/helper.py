import json
import os
import pickle
import random
import re
import shutil
import time
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from PIL import Image

# from werkzeug.datastructures.file_storage import FileStorage
import pymupdf
import io

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


# def log(file_path: str, text: str) -> None:
#     """
#     Send log into log file using append method
#     file_path: str
#     text: str
#     """
#     with open(file_path, "a") as f:
#         f.writelines(f"\n{text}")


# def label_to_price(df_output, verbose=False):
#     """df_output: output of predict_pdf function"""
#     df = df_output.copy()
#     price_label = json.load(open(os.path.join(ROOT_DIR, "models/price_label.json")))
#     price_map = {}
#     for i, price in enumerate(price_label["prices"]):
#         price_map[i] = price
#     df["price"] = df["label"].map(price_map)
#     n_pages_500 = len(df[df["price"] == 500])
#     n_pages_1000 = len(df[df["price"] == 1000])
#     n_pages_1500 = len(df[df["price"] == 1500])
#     n_pages_2000 = len(df[df["price"] == 2000])
#     print(df)

#     if verbose:
#         print(f"""
#             500  x n_pages_500 \t= {500 * n_pages_500}
#             1000 x n_pages_1000 \t= {1000 * n_pages_1000}
#             2000 x n_pages_2000 \t= {2000 * n_pages_2000}
#         """)
#     output = {
#         "items": [
#             (n_pages_500, 500),
#             (n_pages_1000, 1000),
#             (n_pages_1500, 1500),
#             (n_pages_2000, 2000),
#         ],
#         "price_total": sum(df["price"]),
#     }
#     del df
#     return output


# def _extract_images_to_cmyk(
#     file_path_or_pdf_bytes: str | bytes, dpi: int
# ) -> pd.DataFrame:
#     # print(type(file_path_or_pdf_bytes))
#     # if not isinstance(file_path_or_pdf_bytes, bytes):
#     #     raise Exception("FileTypeERROR: File must be a path or pdf file bytes")

#     if isinstance(file_path_or_pdf_bytes, str):
#         if file_path_or_pdf_bytes.split(".")[-1].lower() != "pdf":
#             raise Exception("FileExtensionError: File extension must be pdf")

#     # Check file page type, must be A4
#     # later

#     cmyk = {
#         "c": [],
#         "m": [],
#         "y": [],
#         "k": [],
#     }

#     pdf = pdfium.PdfDocument(file_path_or_pdf_bytes)
#     for i in range(len(pdf)):
#         bitmap = pdf[i].render(
#             scale=1 / 72 * 300,  # 5 DPI
#             rotation=0,
#         )
#         pil_image = bitmap.to_pil()
#         c, m, y, k = sum(calculate_cmyk_percentage(pil_image))
#         cmyk["c"].append(c)
#         cmyk["m"].append(m)
#         cmyk["y"].append(y)
#         cmyk["k"].append(k)

#     return pd.DataFrame(cmyk)


# result = _extract_images_to_cmyk("../datasets/Sample.pdf", 10)

# # def predict_pdf(file_path_or_pdf_bytes):

# #     # Check file type, must be pdf
# #     output = {"time": [], "label": []}
# #     res = sum(calculate_cmyk_percentage(pil_image))
# #     kmeans, scaler = pickle.load(
# #         open(os.path.join(ROOT_DIR, "models", "kmeans_and_scaler.pkl"), "rb")
# #     )
# #     label_pred = kmeans.predict(scaler.transform([pd.Series({"sum": res})]))[0]
# #     print(f"page-{i+1}:", label_pred)

# #     output["time"].append(time.time() - start)
# #     output["label"].append(label_pred)
# #     # del start, bitmap, pil_image, res, kmeans, scaler, label_pred
# #     return pd.DataFrame(output)


# def get_folder_size(folder_path: str) -> float:
#     """
#     Get size of specific folder
#     """
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(folder_path):
#         for filename in filenames:
#             filepath = os.path.join(dirpath, filename)
#             total_size += os.path.getsize(filepath)
#     return round(total_size / (1024 * 1024), 2)  # Mengonversi byte ke MB


# def rmtree(folder_path: str) -> None:
#     try:
#         shutil.rmtree(folder_path)
#         print(f"Folder '{folder_path}' telah dihapus beserta isinya.")
#     except FileNotFoundError:
#         print(f"Folder '{folder_path}' tidak ditemukan.")
#     except PermissionError:
#         print(f"Tidak memiliki izin untuk menghapus folder '{folder_path}'.")
#     except Exception as e:
#         print(f"Terjadi kesalahan: {e}")


# def prepend_zero(file_name):
#     """
#     Prepend string contains number with 0 up to 999
#     example: something-1.jpg -> something-001.jpg
#     """
#     number = int(re.findall("\d+", file_name)[0])
#     result = "page-"
#     if number / 10 < 1:
#         result = f"{result}00{number}.jpg"
#     elif number / 10 < 10:
#         result = f"{result}0{number}.jpg"
#     else:
#         result = f"{result}{number}.jpg"

#     return result


# # Fungsi untuk konversi RGB ke CMYK
# def rgb_to_cmyk(image_path_or_pil_img):
#     # print(type(image_path_or_pil_img) == str, isinstance(image_path_or_pil_img, PIL.Image.Image))
#     if not isinstance(image_path_or_pil_img, str) and not isinstance(
#         image_path_or_pil_img, PIL.Image.Image
#     ):
#         raise Exception(
#             "ImportError: image_path_or_pil_img must be str or PIL.Image.Image, not",
#             type(image_path_or_pil_img),
#         )

#     if isinstance(image_path_or_pil_img, str):
#         if image_path_or_pil_img.split(".")[-1].lower() not in [
#             "jpg",
#             "png",
#             "jpeg",
#             "webp",
#             "heic",
#         ]:
#             raise Exception("FileExtensionError: file type must be an image format")

#     # Membuka gambar dari path dan konversi ke RGB
#     img = None
#     if isinstance(image_path_or_pil_img, str):
#         img = Image.open(image_path_or_pil_img).convert("RGB")

#     else:
#         img = image_path_or_pil_img

#     # Konversi gambar menjadi array NumPy
#     img_array = np.array(img) / 255.0

#     # Konversi RGB ke CMY
#     c = 1 - img_array[:, :, 0]
#     m = 1 - img_array[:, :, 1]
#     y = 1 - img_array[:, :, 2]

#     # Hitung nilai K (Black)
#     k = np.minimum(np.minimum(c, m), y)

#     # Menghindari pembagian oleh nol
#     c = (c - k) / (1 - k + 1e-10)
#     m = (m - k) / (1 - k + 1e-10)
#     y = (y - k) / (1 - k + 1e-10)

#     # Gabungkan dalam array CMYK
#     return np.dstack((c, m, y, k))


# # Fungsi untuk menghitung persentase komponen CMYK
# def calculate_cmyk_percentage(image_path_or_pil_image):
#     # Konversi RGB ke CMYK
#     cmyk_array = rgb_to_cmyk(image_path_or_pil_image)

#     # Menghitung rata-rata persentase dari setiap channel (C, M, Y, K)
#     c_percent = np.mean(cmyk_array[:, :, 0]) * 100
#     m_percent = np.mean(cmyk_array[:, :, 1]) * 100
#     y_percent = np.mean(cmyk_array[:, :, 2]) * 100
#     k_percent = np.mean(cmyk_array[:, :, 3]) * 100

#     # # Output persentase
#     # print(f"Persentase C: {c_percent:.2f}%")
#     # print(f"Persentase M: {m_percent:.2f}%")
#     # print(f"Persentase Y: {y_percent:.2f}%")
#     # print(f"Persentase K: {k_percent:.2f}%")

#     return (
#         round(c_percent, 2),
#         round(m_percent, 2),
#         round(y_percent, 2),
#         round(k_percent, 2),
#     )


# Fungsi menampilkan 8 gambar untuk label tertentu
# def show_clustered_image(df: pd.DataFrame, category, column_name: str = "label"):
#     """
#     Parameters:
#     df      : Pandas DataFrame
#     group   : Group to be showed for respective column_name
#     column_name: Name of a column in the df that used to be showed

#     Usage:
#     show_clustered_image(df, 5) # show a single category based on `label` (default) column, show only category 3

#     for price in df.price.unique():
#         show_clustered_image(df, price, "price") # show images based on "price" column, show all price category
#     """
#     img_index_tobe_showed = list(df[df[column_name] == category].index)

#     # Show 8 images only
#     if len(img_index_tobe_showed) > 8:
#         img_index_tobe_showed = random.sample(img_index_tobe_showed, 8)

#     # Buat subplot
#     fig, ax = plt.subplots(1, len(img_index_tobe_showed))

#     # Jika hanya ada satu gambar, ax tidak berupa array
#     if len(img_index_tobe_showed) == 1:
#         ax = [ax]  # Mengubah ax menjadi list agar bisa diakses dengan indeks

#     for i, img_index in enumerate(img_index_tobe_showed):
#         ax[i].imshow(
#             plt.imread(f"../outputs/pdfium_150dpi/{prepend_zero(str(img_index + 1))}")
#         )
#         ax[i].set_title(f"page-{img_index + 1}")
#         ax[i].axis("off")  # Matikan axis supaya gambar lebih jelas

#     # Menyesuaikan ukuran gambar
#     fig.set_figheight(10)
#     fig.set_figwidth(60)
#     plt.show()


class PrintCost:
    df_dict = {
        "c": [],
        "m": [],
        "y": [],
        "k": [],
        "cmy": [],
        "cmyk": [],
    }

    def __init__(self, file_path: str, model_pkl_path: str) -> None:
        self.file_path = file_path
        self.model_pkl_path = model_pkl_path
        self.model = pickle.load(open(self.model_pkl_path, "rb"))

    def extract_cmyk(self, dpi) -> pd.DataFrame:
        self._reset_df_dict()

        pdf_obj = None

        if isinstance(self.file_path, str):
            pdf_obj = pymupdf.open(self.file_path)
        else:
            pdf_obj = self.file_path

        for page in pdf_obj:
            pixmap = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pixmap.tobytes()))

            cmyk = self._calculate_cmyk_percentage(img)
            self._df_dict_appender(cmyk)

        self.df = pd.DataFrame(self.df_dict)

        return self.df

    def _reset_df_dict(self):
        self.df_dict = {
            "c": [],
            "m": [],
            "y": [],
            "k": [],
            "cmy": [],
            "cmyk": [],
        }

    def _df_dict_appender(self, cmyk):
        c, m, y, k = cmyk
        cmy = c + m + y
        self.df_dict["c"].append(c)
        self.df_dict["m"].append(m)
        self.df_dict["y"].append(y)
        self.df_dict["k"].append(k)
        self.df_dict["cmy"].append(cmy)
        self.df_dict["cmyk"].append(cmy + k)

    def _calculate_cmyk_percentage(self, pil_img):
        # Konversi RGB ke CMYK
        cmyk_array = self._rgb_to_cmyk(pil_img)

        # Menghitung rata-rata persentase dari setiap channel (C, M, Y, K)
        c_percent = np.mean(cmyk_array[:, :, 0]) * 100
        m_percent = np.mean(cmyk_array[:, :, 1]) * 100
        y_percent = np.mean(cmyk_array[:, :, 2]) * 100
        k_percent = np.mean(cmyk_array[:, :, 3]) * 100

        return (
            round(c_percent, 2),
            round(m_percent, 2),
            round(y_percent, 2),
            round(k_percent, 2),
        )

    def _rgb_to_cmyk(self, image_path_or_pil_img):
        # print(type(image_path_or_pil_img) == str, isinstance(image_path_or_pil_img, PIL.Image.Image))
        if not isinstance(image_path_or_pil_img, str) and not isinstance(
            image_path_or_pil_img, PIL.Image.Image
        ):
            raise Exception(
                "ImportError: image_path_or_pil_img must be str or PIL.Image.Image, not",
                type(image_path_or_pil_img),
            )

        if isinstance(image_path_or_pil_img, str):
            if image_path_or_pil_img.split(".")[-1].lower() not in [
                "jpg",
                "png",
                "jpeg",
                "webp",
                "heic",
            ]:
                raise Exception("FileExtensionError: file type must be an image format")

        # Membuka gambar dari path dan konversi ke RGB
        img = None
        if isinstance(image_path_or_pil_img, str):
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

    def predict(self, dpi=7):
        start = time.time()
        self.extract_cmyk(dpi=dpi)
        y_pred = self.model.predict(self.df[["cmy", "k", "cmyk"]])
        self.df["price"] = y_pred
        self.df.price = self.df.price.replace(
            {0: 500, 1: 750, 2: 1000, 3: 1500, 4: 2000}
        )
        response = self._generate_response(start)
        return response

    def _generate_response(self, start_time):
        value_counts = self.df.price.value_counts()
        sub_total = value_counts.index.values * value_counts.values

        result_dict = {
            f"{price}": {"count": count, "sub_total": price * count}
            for price, count in zip(value_counts.index.tolist(), value_counts.values.tolist())
        }
        
        # Tambahkan total dan elapsed_time
        result_dict["total"] = int(sub_total.sum())
        result_dict["elapsed_time"] = time.time() - start_time

        return {"details": result_dict}