"""
A2C Multi-Agent Controller for 3x3 Traffic Grid.

Key difference from MultiAgentController (DQN):
- DQN: learns at every step (off-policy, replay buffer)
- A2C: collects experience during episode, learns ONCE at episode end (on-policy)

The controller collects rollouts; the training script calls `finish_episode()`
at the end of each simulation run to trigger the update.
"""

import numpy as np
import sys
import os

# Ensure src.core is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core import rl_utils


class A2CMultiAgentController:
    """Controls 9 A2C agents on a 3x3 intersection grid."""

    def __init__(
        self,
        ts_ids,
        ts_info,
        agents_dict,
        training: bool = True,
        reward_weights=None,
        min_phase: int = 10,
        decision_interval: int = 5,
        exploration_epsilon: float = 0.0,
        global_reward_weight: float = 0.05,
    ):
        self.ts_ids              = ts_ids
        self.ts_info             = ts_info
        self.agents              = agents_dict   # ts_id → A2CAgent
        self.training            = training
        self.reward_weights      = reward_weights or {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.1}
        self.exploration_epsilon = exploration_epsilon
        self.global_reward_weight= global_reward_weight
        self.min_phase           = min_phase
        self.decision_interval   = decision_interval

        # State tracking per agent
        self.phase_state          = {}
        self.last_decision_time   = {}
        self.prev_step_info       = {}   # ts_id → (state, action) | None

        for ts_id in ts_ids:
            self.phase_state[ts_id] = {
                'current_phase': 0,
                'time_in_phase': 0,
                'is_yellow': False,
            }
            self.last_decision_time[ts_id] = 0
            self.prev_step_info[ts_id]     = None

        self._local_wait_cache = {}

    # ------------------------------------------------------------------
    # Called every simulation step by engine.py
    # ------------------------------------------------------------------
    def step(self, traci_conn, global_metrics):
        current_time = traci_conn.simulation.getTime()

        for ts_id in self.ts_ids:
            # 1. Update phase state
            try:
                curr_phase_idx = traci_conn.trafficlight.getPhase(ts_id)
                prev_phase     = self.phase_state[ts_id]['current_phase']
                if curr_phase_idx != prev_phase:
                    self.phase_state[ts_id]['current_phase'] = curr_phase_idx
                    self.phase_state[ts_id]['time_in_phase'] = 0
                    self.phase_state[ts_id]['is_yellow']     = (curr_phase_idx % 2 == 1)
                else:
                    self.phase_state[ts_id]['time_in_phase'] += 1
            except Exception:
                continue

            # 2. Skip yellow or too-soon decisions
            if self.phase_state[ts_id]['is_yellow']:
                continue
            if self.phase_state[ts_id]['time_in_phase'] < self.min_phase:
                continue
            if (current_time - self.last_decision_time[ts_id]) < self.decision_interval:
                continue

            # 3. Decision point
            self.last_decision_time[ts_id] = current_time
            agent  = self.agents[ts_id]
            state  = rl_utils.get_state(None, self.phase_state, ts_id, self.ts_info, traci_conn=traci_conn)

            # 4. Compute reward for PREVIOUS action
            if self.prev_step_info[ts_id] is not None:
                prev_state, prev_action = self.prev_step_info[ts_id]
                local_metrics = rl_utils.get_local_metrics(traci_conn, ts_id, self.ts_info, self._local_wait_cache)

                if isinstance(self.reward_weights, dict) and ts_id in self.reward_weights and isinstance(self.reward_weights[ts_id], dict):
                    agent_weights = self.reward_weights[ts_id]
                else:
                    agent_weights = self.reward_weights

                base_reward   = rl_utils.compute_reward(local_metrics, None, agent_weights)
                global_inc_w  = global_metrics.get('incremental_waiting_time', 0.0)
                lambda_g      = agent_weights.get('lambda_global', self.global_reward_weight) if isinstance(agent_weights, dict) else self.global_reward_weight
                reward        = base_reward - (lambda_g * global_inc_w)

                if self.training:
                    agent.store_transition(prev_state, prev_action, reward, state, False)
                    # NOTE: agent.learn() is a no-op for A2C; update happens at episode end

            # 5. Select action
            if not self.training:
                if self.exploration_epsilon > 0:
                    action = agent.select_action(state, training=True, epsilon=self.exploration_epsilon)
                else:
                    action = agent.select_action(state, training=False)
            else:
                action = agent.select_action(state, training=True)

            # 6. Apply action
            curr_phase_idx = traci_conn.trafficlight.getPhase(ts_id)
            if action == 1:
                num_phases = self.ts_info[ts_id]['num_phases']
                next_phase = (curr_phase_idx + 1) % num_phases
                traci_conn.trafficlight.setPhase(ts_id, next_phase)
                self.phase_state[ts_id]['time_in_phase'] = 0
            else:
                traci_conn.trafficlight.setPhase(ts_id, curr_phase_idx)

            self.prev_step_info[ts_id] = (state, action)

        # Update local wait cache
        all_vehs = traci_conn.vehicle.getIDList()
        self._local_wait_cache = {v: traci_conn.vehicle.getAccumulatedWaitingTime(v) for v in all_vehs}


    # ------------------------------------------------------------------
    # Call this at the END of every episode to trigger A2C updates
    # ------------------------------------------------------------------
    def finish_episode(self):
        """Triggers learn_episode() for every agent and returns avg loss."""
        losses = []
        for ts_id, agent in self.agents.items():
            loss = agent.learn_episode(last_value=0.0) # V(s_Terminal)=0
            if loss is not None:
                losses.append(loss)
        return float(np.mean(losses)) if losses else 0.0
