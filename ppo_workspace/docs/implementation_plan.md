# DQN Integration Plan

## Goal
Integrate a Deep Q-Network (DQN) agent to control the traffic light phases, replacing the fixed/rule-based logic.

## 1. New Files
*   `src/core/rl_agent.py`: Contains `DQNAgent`, `ReplayBuffer`, and `DQN` (Neural Network) classes.
*   `src/core/rl_utils.py`: Helper functions for:
    *   `get_state(traci_conn, ts_id)`: Extracts feature vector.
    *   `compute_reward(metrics, weights)`: Calculates scalar reward.
*   `src/core/rl_controllers.py`:
    *   `RLController`: Inherits from `BaseController`. Interacts with `DQNAgent` to make decisions (Action 0=Keep, 1=Switch). Handles experience storage (if training).
*   `scripts/train_dqn.py`: Main training loop (Runs episodes, calls engine, saves models).
*   `scripts/run_dqn_eval.py`: Evaluation script (Runs 5 seeds with greedy policy, outputting compatible CSVs).

## 2. Approach
*   **State Space**: `[avg_queue, avg_wait, avg_speed, phase_idx, time_in_phase]` (Size: 5).
*   **Action Space**: Discrete(2) -> `{0: Keep, 1: Switch}`.
*   **Reward**: Linear combination of negative metrics (minimizing cost).
*   **Logic**:
    *   Controller checks if `min_phase_duration` has passed.
    *   If yes, queries Agent.
    *   Agent returns action.
    *   Controller executes action.

## 3. Verification Plan
*   **Sanity Check**: Verify `train_dqn.py` runs for 1 episode without crashing.
*   **Evaluation Check**: Run `scripts/run_dqn_eval.py` and verify `results/dqn/seed_X/realtime_metrics.csv` are generated and valid.
*   **Performance**: Compare DQN summary metrics against Baseline to ensure it's behaving reasonably (even if not optimal yet).
