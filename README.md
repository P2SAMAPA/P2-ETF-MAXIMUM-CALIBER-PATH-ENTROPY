# Maximum Caliber – Path Entropy Engine for ETFs

Applies the principle of maximum caliber (path entropy maximisation) to ETF return paths. Computes the **entropy rate** (average uncertainty per step) of discretised returns, which measures path predictability.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Discretisation into N bins (e.g., down, flat, up)
- First‑order Markov chain for entropy rate estimation
- Score = entropy rate (higher = less predictable)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-maximum-caliber-path-entropy-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High path entropy → ETF returns are unpredictable, chaotic, potentially regime‑shifting.
- Low path entropy → structured, trending, predictable.

## Requirements

See `requirements.txt`.
