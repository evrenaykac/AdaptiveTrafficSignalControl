# When Do LLM Coaches Help? An Auditable, Episodic-Memory Framework for RL Reward Shaping in Adaptive Traffic Signal Control

This repository accompanies our IEEE Access submission on using a **locally hosted large language model (LLM)** as an *auditable* coach that tunes the reward weights of a reinforcement-learning (RL) traffic-signal controller, optionally informed by a **Retrieval-Augmented Generation (RAG)** memory of past training episodes. It contains the manuscript (LaTeX source + compiled PDF), the confirmed experimental results, and a detailed plan for the follow-up experiments.

> **Honest one-line summary.** Adaptive RL clearly beats the existing fixed-time plan at our study intersection, but across four RL algorithms we find **no statistically significant benefit** from LLM or RAG reward-coaching at five seeds — so we present this as a *cautionary, "when-does-it-help"* study plus a reusable, safety-constrained framework, not as a "coaching wins" result.

---

## Study at a glance

- **Problem.** Deep-RL adaptive traffic signal control works well, but the multi-objective reward (waiting time, queue, emissions) is shaped by hand and brittle.
- **Idea.** Let an LLM propose new reward weights between training episodes, *bounded by a deterministic safety layer*, and optionally let it retrieve semantically similar past episodes (RAG) before deciding.
- **Question.** Does LLM coaching — with or without episodic memory — actually help, and for which algorithms?
- **Test bed.** A SUMO digital twin of the Halil Sezai Erkut Avenue intersection (Etlik, Ankara, Türkiye), built from on-site demand counts, with a high-fidelity multi-lane connection model.
- **Design.** 4 RL algorithms (DQN, DDPG, PPO, A2C) × 4 conditions (Fixed-time, Standard RL, +LLM coach, +LLM+RAG coach), 5 seeds per cell.
- **LLM.** Llama 3.1 8B, served locally via Ollama (no external API) — privacy-preserving and auditable.

---

## Key findings (confirmed data)

1. **Adaptive RL is effective.** All 12 learning-based configurations beat the fixed-time baseline (108.0 s mean waiting time). Best configuration **A2C + PURE**: 86.53 s (**−19.9%** waiting), **+29.3%** speed (32.1 → 41.5 km/h), **−17.6%** CO₂ rate (185.4 → 152.7 g/km), **−43.0%** queue.
2. **Coaching shows no significant effect.** Across all eight *PURE-vs-LLM* and *LLM-vs-RAG* contrasts, **none reaches statistical significance** after Holm–Bonferroni correction (smallest Holm *p* = 0.25). Effect sizes are small-to-negligible.
3. **The best overall configuration uses no coaching at all** (A2C + PURE).
4. **Power caveat.** With five seeds the power to detect small/medium effects is low (Wilcoxon two-sided floor at *n*=5 is *p*=0.0625). Non-significant results mean *absence of evidence for a large effect*, not *evidence of no effect* — see [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) for the properly powered follow-up.

The full numbers are in [`data/results_per_metric.csv`](data/results_per_metric.csv) and [`data/results_stats.csv`](data/results_stats.csv), and as Tables 5–6 in the paper.

---

## Repository structure

```
.
├── README.md                     # this file
├── LICENSE                       # MIT (scope: code/scripts) — see "License"
├── CITATION.cff                  # how to cite this work
├── .gitignore                    # LaTeX build artifacts
│
├── paper/                        # self-contained IEEE Access manuscript
│   ├── access.tex                #   main LaTeX source
│   ├── access.pdf                #   compiled manuscript
│   ├── ubmk.bib                  #   bibliography (55 references)
│   ├── access.bbl                #   pre-built bibliography
│   ├── ieeeaccess.cls, IEEEtran.cls, IEEEtran.bst, spotcolor.sty
│   ├── *.tfm *.fd *.pfb *.map     #   IEEE Access template fonts (for offline build)
│   ├── Diagram.png, Junction_Connections.png, RAGDiagram.png
│   ├── results_waiting.png        #   Fig. 6 (generated from the confirmed results)
│   └── photo_*.png                #   author photo placeholders (replace before submission)
│
├── docs/
│   └── EXPERIMENT_DESIGN.md       # plan for the strengthening experiments (baselines, ablations, stats)
│
├── data/
│   ├── README.md
│   ├── results_per_metric.csv     # waiting/queue/speed/CO2 mean±sd (+CI) for all 16 cells
│   └── results_stats.csv          # paired Wilcoxon contrasts, Cohen's d, Holm p
│
└── code/
    └── README.md                  # placeholder — add the SUMO + RL + LLM-coach code here
```

> **Note on `data/` and `code/`.** The simulation environment, the NumPy RL agents, the LLM-coach loop, and the raw per-seed logs live on the authors' machines and should be added under `code/` and `data/` before the repository is made public. The placeholders describe the expected contents.

---

## The paper

**Title.** *When Do LLM Coaches Help? An Auditable, Episodic-Memory Framework for Reinforcement Learning Reward Shaping in Adaptive Traffic Signal Control.*

### Build it yourself
The `paper/` folder is self-contained (the IEEE Access class and required fonts are vendored), so a clean LaTeX install can build it offline:

```bash
cd paper
pdflatex access.tex
bibtex   access
pdflatex access.tex
pdflatex access.tex
```

This produces `access.pdf` (13 pages). On Overleaf, upload the contents of `paper/` and compile with pdfLaTeX; the vendored fonts are harmless there.

---

## Method summary

- **MDP.** State = (max queue, cumulative waiting, mean speed, phase index, phase elapsed), all normalised; action ∈ {keep, advance}; reward `R = −αW − βQ − γCO₂` with `α+β+γ=1`.
- **Coach.** After each episode, a structured summary is embedded (sentence-transformer) and stored in a FAISS index with its weights. Before the next episode the coach retrieves the top-*k* similar past episodes (RAG), prompts Llama 3.1 8B for new weights, and logs the rationale.
- **Safety layer (the auditability core).** Every LLM proposal is (i) re-normalised to the simplex, (ii) clipped to ±0.10 per weight per episode, (iii) EMA-smoothed (0.8 old / 0.2 new), and (iv) JSON-sanitised — so the coach's influence is bounded, reversible, and logged.
- **Conditions.** `PURE` = fixed weights, no LLM; `+LLM` = coach with `k=0` (no memory); `+RAG` = full coach with `k=3`.

See the paper (Sections III–IV) and [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) for full detail.

---

## Reproducing & extending

The current results use five seeds; the [experiment-design plan](docs/EXPERIMENT_DESIGN.md) specifies the experiments that would make the conclusions decisive, in priority order:

1. **Strong adaptive baselines** — max-pressure and actuated control (beating a static plan is a low bar).
2. **Control coaches** — replace the LLM with *random-in-safety-layer* and *rule-based* proposals (same safety layer) to test whether the LLM adds anything beyond the safety layer.
3. **Grid-search ceiling** — the best fixed weights per algorithm, to check whether there is anything for a coach to find.
4. **More seeds (≥15–20) + proper statistics** — Wilcoxon + bootstrap CIs + effect sizes + Holm correction.
5. **Ablations** — retrieval depth *k*, EMA coefficient, safety cap, temperature.
6. **Cost/overhead** — wall-clock and energy of the local LLM.

---

## Authors

- **Yusuf Evren Aykaç**¹ — *first author* — Lecturer; postdoctoral researcher on a TÜBİTAK-funded project
- **Faruk Büyüktekin**² — *corresponding author*
- **Onur Çokyiğit**¹, **Arif Özalp**¹, **Meriç Uysalerler**¹, **Mert Bilgiç**¹

¹ Department of Computer Engineering, Ankara Yıldırım Beyazıt University (AYBU), Ankara, Türkiye
² Cognitive Science, Middle East Technical University (METU), Ankara, Türkiye

> Corresponding-author e-mail and the co-author biographies/photos are placeholders in the current manuscript and should be completed before submission.

---

## Citation

If you use this work, please cite the paper (see [`CITATION.cff`](CITATION.cff)). A provisional BibTeX entry:

```bibtex
@article{aykac_llm_coach_atsc,
  title   = {When Do {LLM} Coaches Help? An Auditable, Episodic-Memory Framework
             for Reinforcement Learning Reward Shaping in Adaptive Traffic Signal Control},
  author  = {Ayka{\c{c}}, Yusuf Evren and B{\"u}y{\"u}ktekin, Faruk and
             {\c{C}}okyi{\u{g}}it, Onur and {\"O}zalp, Arif and
             Uysalerler, Meri{\c{c}} and Bilgi{\c{c}}, Mert},
  journal = {(under review, IEEE Access)},
  year    = {2026}
}
```

---

## Acknowledgments

This work was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) under the 2209-A Research Project Support Programme.

---

## License

The code and scripts in this repository are released under the [MIT License](LICENSE).

The manuscript text and figures are © the authors; upon publication the article will be governed by IEEE's copyright policy, so please consult that policy before redistributing the PDF. The result datasets (`data/*.csv`) are released under CC BY 4.0.
