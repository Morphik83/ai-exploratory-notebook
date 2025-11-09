# AI Exploratory Notebook

A minimal demo app (Flask) and an AI-assisted exploratory testing tool that captures pre/post UI states, computes visual/DOM diffs, and generates a one-page insight report using an LLM.

## Quick start

1. Create venv and install deps (PowerShell):
   - `python -m venv .venv`
   - `.venv\\Scripts\\Activate.ps1`
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`
   - `python -m playwright install`

2. Run demo app:
   - `python demo_app/app.py`
   - App runs at http://localhost:5000

3. Run exploratory tool (single page):
   - Home variant diff:
     - `python tools/exploratory_notebook.py --pre-url http://localhost:5000/?variant=A --post-url http://localhost:5000/?variant=B --out reports --provider openai --model gpt-4o-mini --scenario home`
   - Products variant diff:
     - `python tools/exploratory_notebook.py --pre-url http://localhost:5000/products?variant=A --post-url http://localhost:5000/products?variant=B --out reports --provider openai --model gpt-4o-mini --scenario products`

4. Run a batch with YAML (multi-page):
   - Create `scenarios/demo.yaml`:
     ```yaml
     runs:
       - name: home
         pre_url: "http://localhost:5000/?variant=A"
         post_url: "http://localhost:5000/?variant=B"
       - name: products
         pre_url: "http://localhost:5000/products?variant=A"
         post_url: "http://localhost:5000/products?variant=B"
     ```
   - Execute:
     - `python tools/exploratory_notebook.py batch --scenario-file scenarios/demo.yaml --out reports --provider openai --model gpt-4o-mini`
   - Open `reports/index.html` to navigate to each report.

Notes:
- Set `OPENAI_API_KEY` in a local `.env` at repo root (auto-loaded) or as an environment variable.
- Reports and `.env` are git-ignored by default.
