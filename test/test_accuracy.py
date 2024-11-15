import os, sys
import joblib
import json
import pandas as pd

sys.path.append(os.path.abspath(".."))
from src.helper import calculate_cmyk_percentage, ROOT_DIR
import pypdfium2 as pdfium

model, scaler = joblib.load(open(os.path.join(ROOT_DIR, 'models', 'kmeans_and_scaler.pkl'), 'rb'))
labels = json.load(open(os.path.join(ROOT_DIR, 'models', 'price_label.json')))

def render_and_predict(page, dpi:int) -> float:
  """Render a single page pillow image"""
  bitmap = page.render(
            scale = 1/72 * dpi, # 5 DPI
            rotation = 0, 
        )
  pil_image = bitmap.to_pil()
  cmyk_sum = sum(calculate_cmyk_percentage(pil_image))
  
  index_label = model.predict(scaler.transform([[cmyk_sum]]))[0]
  return labels['prices'][index_label]

df_json = {
  "filename": [],
  "page": [],
  "res300": [],
  "res50": [],
  "res5": []
}

for doc_file in os.listdir(os.path.join(ROOT_DIR, 'datasets', 'test-set')):
  curr_dir = os.path.join(ROOT_DIR, 'datasets', 'test-set')
  if doc_file.split(".")[-1] == 'pdf':
    doc = pdfium.PdfDocument(os.path.join(curr_dir, doc_file))
    for i, page in enumerate(doc):
      res300 = render_and_predict(page, 300)
      res50 = render_and_predict(page, 50)
      res5 = render_and_predict(page, 5)
      
      df_json['filename'].append(doc_file)
      df_json['page'].append(i+1)
      df_json['res300'].append(res300)
      df_json['res50'].append(res50)
      df_json['res5'].append(res5)
  print(doc_file)

df = pd.DataFrame(df_json)
df.to_csv(os.path.join(ROOT_DIR, 'outputs', 'dpi_300_50_5_accuracy.csv'), index=False)