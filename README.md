---
title: Print Cost
emoji: 🖨
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
short_description: A simple tool calculate printing costs
---

# print-cost

PrintCost is a simple tool designed to calculate printing costs for A4 paper.

Upload a PDF and it renders each page at 7 DPI, converts it to CMYK, computes the
average ink coverage per channel (`cmy`, `k`, `cmyk`), and classifies each page
with a trained XGBoost model into one of five colour-intensity buckets. The
buckets map to rupiah per page:

| Label | Price / A4 page | Category |
|---|---|---|
| 0 | Rp 500 | Mono Print |
| 1 | Rp 750 | Color Light |
| 2 | Rp 1.000 | Color Standard |
| 3 | Rp 1.500 | Color Heavy |
| 4 | Rp 2.000 | Full Color – Dark & Mixed |

## What this prices, and what it does not

- **It prices one thing:** A4 pages in those five buckets. The buckets are a
  business decision, not a market survey — the model's job is only to decide
  which bucket a page falls into.
- **It does not price** laminating, binding/jilid, finishing, paper upgrades or
  quantities of anything other than A4 pages.
- **The 7 DPI render is intentional.** It is the resolution the model was trained
  and validated at; a higher DPI is not "more accurate", it is out of contract.

## The price table has exactly one definition

`src/config.py` holds the single `PRICE_TABLE`; `PRICE_LABEL_MAP`,
`PRICE_CATEGORY_MAP` and `CATEGORY_COLOR_MAP` are derived from it, and the
Streamlit UI imports the colour map instead of copying it.
`tests/unit/test_price_table.py` asserts that the table has five entries mapping
to the model's five classes and that **no other file in the repository**
re-declares the prices.

## Model artifact

The classifier is stored in Git LFS. A clone without `git lfs pull` contains a
~130 byte pointer, so both the loader and the test suite refuse to treat a
pointer as a model.

```bash
git lfs install
git lfs pull                                        # normal local development
python scripts/fetch_model.py --check               # verify against the pin
python scripts/fetch_model.py                       # or download from the Space
```

`src/config.py` pins the artifact's URL, size (409,239 bytes) and SHA-256
(`523ff4dd…73c46`); `scripts/fetch_model.py` refuses to write anything that does
not match, and `ModelManager.get_default_model()` verifies the local file before
loading it. The pickled model unpickles only with `numpy==1.24.3`,
`scikit-learn==1.3.0` and `xgboost==2.1.2` — do not bump them.

## Pricing a PDF from bytes

For in-process callers (e.g. an agent tool that receives an upload) there is a
bytes entry point that shares one validation path with the HTTP API:

```python
from src.services.cost_calculator import calculate_cost_from_bytes

result = calculate_cost_from_bytes(pdf_bytes, dpi=7)
# {'total_pages': 3, 'total_price': 2500, 'details': [...], 'processing_time': 0.02}
```

It enforces the 50 MB size cap, a `%PDF` header check and the 1–1000 page range,
and raises typed exceptions (`InvalidPDFError`, `FileSizeError`,
`PageCountError`) instead of returning a guessed price.

## Development

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt pytest pytest-cov pytest-mock httpx
.venv/bin/python -m pytest          # pytest.ini enforces --cov-fail-under=70
```

CI (`.github/workflows/ci.yml`) runs the same suite on every push to `main` and
every pull request, pulls the pinned model artifact, and verifies its SHA-256.

## Deployment

The Space runs the Docker image built from this repository
(`streamlit run main.py`, port 8501). To sync the repository to the Space:

```bash
export HF_TOKEN=hf_...                 # write access to the Space
bash scripts/deploy_space.sh           # dry run: shows what would change
bash scripts/deploy_space.sh --apply   # push
```

**Known drift:** the currently deployed Space still runs pre-refactor code and
serves Streamlit only, so `/health` and `/openapi.json` return the Streamlit HTML
page rather than JSON. Serving the FastAPI app instead means changing the
Dockerfile entrypoint, which changes what visitors see; the deploy script
documents both options and does neither automatically.
