import pypdfium2 as pdfium
import PIL
import os, sys, time
import pandas as pd
import logging

# Konfigurasi logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.helper import calculate_cmyk_percentage, log



def generate_dataset(output_path, **render_options):
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'statistik-indonesia-2024-combined.pdf',)
    pdf = pdfium.PdfDocument(pdf_path)
    
    pdf_length = len(pdf)
    logging.info(f"Pdf file loaded, it has {pdf_length} pages, starting the process")
    
    base_df = {
        'converting_time': [],
        'page': [],
        'dpi': [],
        'c': [],
        'm': [],
        'y': [],
        'k': [],
        'sum': [],
        'base_minus_curr_sum': [],
        'diff': [], # diff from the base dpi (300 dpi) in jpg format, no additional setting
    }

    for dpi in list(range(300, 4, -10)) + [5]:
        logging.info(f"start process for {dpi} dpi")
        
        for i in range(pdf_length):
            start = time.time()
            bitmap = pdf[i].render(
                scale = 1/72 * dpi,
                **render_options
            )
            pil_image = bitmap.to_pil()
            converting_time = time.time() - start
            c, m, y, k = calculate_cmyk_percentage(pil_image)
            sum_of_cmyk = c + m + y + k
            base_minus_curr_sum = 0
            diff = 0
            if dpi != 300:
                base_minus_curr_sum = base_df['sum'][i] - sum_of_cmyk
                diff = abs(base_df['sum'][i] - sum_of_cmyk)
            
            # Set 300 dpi as a base DataFrame
            if dpi == 300:
                base_df['converting_time'].append(converting_time)
                base_df['page'].append(i+1)
                base_df['dpi'].append(dpi)
                base_df['c'].append(c)
                base_df['m'].append(m)
                base_df['y'].append(y)
                base_df['k'].append(k)
                base_df['sum'].append(sum_of_cmyk)
                base_df['base_minus_curr_sum'].append(0)
                base_df['diff'].append(0)
            
            if not os.path.exists(os.path.join('datasets', output_path)):
                df = pd.DataFrame(base_df)
                df.to_csv(os.path.join('datasets', output_path), index=False)
                
            else:
                # Append out_path
                log(os.path.join('datasets', output_path), f"{converting_time},{i+1},{dpi},{c},{m},{y},{k},{sum_of_cmyk},{base_minus_curr_sum},{diff}")
    
            # Clean up memory
            del start, bitmap, pil_image, c, m, y, k, sum_of_cmyk, base_minus_curr_sum
            
            # logging.info each multiply by 100
            if i%100 == 0:
                logging.info(f"converted {i+1} page")
            
        logging.info(f"finish process for {dpi} dpi")

if __name__ == '__main__':
    # Options: no_smoothtext (bool), no_smoothimage (bool)
    generate_dataset("cmyk_by_dpi_and_render_options.csv", no_smoothtext=False, no_smoothimage=False)
    generate_dataset("cmyk_by_dpi_and_render_options.csv", no_smoothtext=True, no_smoothimage=False)
    generate_dataset("cmyk_by_dpi_and_render_options.csv", no_smoothtext=True, no_smoothimage=True)