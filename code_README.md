# Code

Add the implementation here before publishing. Based on the paper, the expected
components are:

- **`env/`** — the SUMO/TraCI environment wrapper (state extraction, action
  application, reward computation including the HBEFA3 CO₂ term).
- **`agents/`** — the four RL agents (DQN, DDPG, PPO, A2C), implemented in NumPy
  with a shared 2×64 MLP backbone.
- **`coach/`** — the episodic-memory LLM coach: episode summarisation,
  sentence-transformer embedding, FAISS retrieval, the Ollama (Llama 3.1 8B)
  prompt, and the deterministic **safety layer** (simplex re-normalisation,
  ±0.10 clip, EMA 0.8/0.2, JSON sanitisation).
- **`baselines/`** — fixed-time, and (per the experiment plan) max-pressure and
  actuated controllers, plus the random-in-safety-layer and rule-based control
  coaches.
- **`analysis/`** — the statistics pipeline (Wilcoxon signed-rank, bootstrap CIs,
  Cohen's *d* / Cliff's δ, Holm–Bonferroni) that produces `data/results_*.csv`.
- **`run.py` / `configs/`** — entry point and per-condition configuration; pin the
  exact Ollama model build (e.g. `llama3.1:8b-instruct-q4_K_M`) and the seed set.

A `requirements.txt` (sumo/traci, numpy, faiss-cpu, sentence-transformers,
scipy, statsmodels, matplotlib, ollama) should accompany the code.

See `../docs/EXPERIMENT_DESIGN.md` for the planned experiments and a statistics
script skeleton.
