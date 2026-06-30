# LLM Reward-Coaching for Traffic Signal Control

An RL traffic-signal controller for the Etlik intersection in Ankara, where a
local LLM (Ollama, Llama 3.1 8B) re-tunes the reward weights between training
episodes. The LLM coach is compared against simple **random** and **rule**
controls, across four algorithms (DQN, DDPG, PPO, A2C).

## Install
- Install **SUMO** and set `SUMO_HOME` — https://sumo.dlr.de
- Run **Ollama** with the model: `ollama pull llama3.1:8b`
- `pip install -r requirements.txt`

## Run
```bash
cd SumoMainFunctions
python dqn_workspace/scripts/train_dqn_single.py        # standard / LLM-coached
python dqn_workspace/scripts/train_dqn_rag_single.py    # with RAG memory
```
Same pattern for `ddpg`, `ppo`, `a2c`. Metrics are written to each `*/results/` folder.

## Layout
- `SumoMainFunctions/src/` — engine, RL agents, LLM reward tuner, RAG memory
- `SumoMainFunctions/<algo>_workspace/` — per-algorithm agent + training scripts
- `etlik_intersection/` — SUMO scenario (the study intersection)
