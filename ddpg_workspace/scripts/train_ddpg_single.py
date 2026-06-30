import sys
import os
import argparse
import csv
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core import engine, utils, rl_utils
from src.core.controllers import BaseController
from ddpg_workspace.core.ddpg_agent import DDPGAgent
from src.core.llm_multiagent_reward_tuner import LLMRewardTuner

class SingleDDPGController(BaseController):
    def __init__(self, ts_ids, traffic_signals_info, phase_duration=40, agent=None, training=False, reward_weights=None):
        super().__init__(ts_ids, traffic_signals_info, phase_duration)
        self.agent = agent
        self.training = training
        self.weights = reward_weights if reward_weights else {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.1}
        self.ts_id = ts_ids[0]
        self.last_state = None
        self.last_action = None
        self.min_green = 10
        
    def step(self, traci_conn, metrics):
        self.phase_state[self.ts_id]['time_in_phase'] += 1
        
        if self.phase_state[self.ts_id]['time_in_phase'] < self.min_green:
            return
            
        current_state = rl_utils.get_state(metrics, self.phase_state, self.ts_id, self.ts_info, traci_conn)
        local_metrics = rl_utils.get_local_metrics(traci_conn, self.ts_id, self.ts_info)
        
        raw_m = {
            'queue': local_metrics['avg_queue_length'],
            'wait': local_metrics['incremental_waiting_time'] if local_metrics['incremental_waiting_time'] > 0 else local_metrics['total_accumulated_waiting_time']/100.0,
            'co2': local_metrics['co2_emission_kg_per_s']
        }

        if self.last_state is not None and self.training:
            # Store tuple into Dynamic Replay Buffer 
            self.agent.store_transition(self.last_state, self.last_action, raw_m, current_state, False)
            # DDPG evaluates dynamic reward on sampling inside the `learn` function
            self.agent.learn(self.weights)
            
        action = self.agent.select_action(current_state, training=self.training)
        
        if action == 1:
            curr_phase_idx = self.phase_state[self.ts_id]['current_phase']
            num_phases = self.ts_info[self.ts_id]['num_phases']
            new_phase = (curr_phase_idx + 1) % num_phases
            traci_conn.trafficlight.setPhase(self.ts_id, new_phase)
            self.phase_state[self.ts_id]['current_phase'] = new_phase
            self.phase_state[self.ts_id]['time_in_phase'] = 0
            
        self.last_state = current_state
        self.last_action = action

def run_experiment(mode, out_file, episodes=10, duration=3600, use_gui=False):
    sumocfg_file = str(PROJECT_ROOT.parent / "Etlik tek kavşak" / "osm.sumocfg")
    
    agent = DDPGAgent(state_dim=5, action_dim=2, epsilon=1.0) if mode != "baseline" else None
    tuner = LLMRewardTuner(timeout=60, max_delta=0.10) if mode == "ddpg_llm" else None
    
    current_weights = {'alpha': 0.6, 'beta': 0.3, 'gamma': 0.1}
    
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'mode', 'mean_wait', 'mean_queue', 'total_co2', 'mean_speed', 'alpha', 'beta', 'gamma'])
    
    for ep in range(1, episodes + 1):
        if mode == "baseline":
            from src.core.controllers import FixedTimeController
            ctrl_class = FixedTimeController
            ctrl_kwargs = {}
        else:
            ctrl_class = SingleDDPGController
            ctrl_kwargs = {"agent": agent, "training": True, "reward_weights": current_weights}
            
        print(f"\n--- Phase: {mode} | Episode {ep}/{episodes} ---")
        metrics_history = engine.run_simulation(
            sumocfg_file=sumocfg_file, num_seconds=duration,
            controller_class=ctrl_class, controller_kwargs=ctrl_kwargs,
            use_gui=use_gui, seed=ep
        )
        
        df = pd.DataFrame(metrics_history)
        mean_wait = df['avg_waiting_time_per_vehicle'].mean() if 'avg_waiting_time_per_vehicle' in df.columns else 0.0
        mean_q = df['avg_queue_length'].mean() if 'avg_queue_length' in df.columns else 0.0
        total_co2 = df['co2_emission_kg_per_s'].sum() if 'co2_emission_kg_per_s' in df.columns else 0.0
        mean_speed = df['avg_speed_m_per_s'].mean() * 3.6 if 'avg_speed_m_per_s' in df.columns else 0.0
        
        print(f"Results -> Wait: {mean_wait:.2f}s | Queue: {mean_q:.2f} | CO2: {total_co2:.2f} | Speed: {mean_speed:.2f}")
        
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, mode, mean_wait, mean_q, total_co2, mean_speed, current_weights['alpha'], current_weights['beta'], current_weights['gamma']])

        if agent:
            agent.update_epsilon()

        if tuner:
            print("Running LLM Tuner...")
            global_metrics = {
                "mean_waiting_time": round(mean_wait, 2), "mean_queue": round(mean_q, 2),
                "total_co2": round(total_co2, 2), "mean_speed": round(mean_speed, 2)
            }
            local_metrics_by_tls = {'t': global_metrics}
            new_weights_dict, meta = tuner.tune(global_metrics, local_metrics_by_tls, {"t": current_weights})
            current_weights = new_weights_dict.get('t', current_weights)
            # Memory flushing is fully DEPRECATED.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "ddpg", "ddpg_llm"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=int, default=1500)
    parser.add_argument("--gui", action="store_true", help="Run with SUMO GUI")
    args = parser.parse_args()
    
    utils.check_sumo_env()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    run_experiment(args.mode, args.out, args.episodes, args.duration, args.gui)
