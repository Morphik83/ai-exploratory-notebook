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

3. Run exploratory tool:
   - `python tools/exploratory_notebook.py --pre-url http://localhost:5000/?variant=A --post-url http://localhost:5000/?variant=B --out reports --provider openai --model gpt-4o-mini`

Set `OPENAI_API_KEY` to enable LLM insights. Without it, the tool still captures and diffs.
