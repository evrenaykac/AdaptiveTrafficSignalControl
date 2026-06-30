"""
PPO Agent for Multi-Agent Traffic Signal Control.
Uses a NumPy-based Actor-Critic network to match the existing DQN stack (no PyTorch).
Interface is intentionally compatible with DQNAgent so MultiAgentController can be reused.
"""

import random
import numpy as np
import pickle
from collections import deque


# ---------------------------------------------------------------------------
# Utility: Softmax
# ---------------------------------------------------------------------------
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Simple Actor-Critic MLP (NumPy, no PyTorch dependency)
# ---------------------------------------------------------------------------
class ActorCriticMLP:
    """
    Shared-body network with two heads:
      - Actor head  → action logits  (shape: action_dim)
      - Critic head → state value    (shape: 1)
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64, lr: float = 3e-4):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Shared body
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        # Actor head
        self.W_actor = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_actor = np.zeros((1, action_dim))

        # Critic head
        self.W_critic = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b_critic = np.zeros((1, 1))

    # ---- Forward ----
    def forward(self, X: np.ndarray):
        """Returns (action_probs, state_values)."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)                          # ReLU
        logits = np.dot(a1, self.W_actor) + self.b_actor
        probs  = softmax(logits)                         # (batch, action_dim)
        values = np.dot(a1, self.W_critic) + self.b_critic  # (batch, 1)
        return probs, values.squeeze(-1), a1, z1

    def get_value(self, X: np.ndarray) -> np.ndarray:
        _, values, _, _ = self.forward(X)
        return values

    # ---- Update (SGD) ----
    def update(self,
               states: np.ndarray,
               actions: np.ndarray,
               returns: np.ndarray,
               advantages: np.ndarray,
               old_log_probs: np.ndarray,
               clip_eps: float = 0.2,
               entropy_coef: float = 0.01,
               value_coef: float = 0.5):
        """One gradient step of PPO clipped surrogate loss."""
        probs, values, a1, z1 = self.forward(states)
        m = states.shape[0]

        # --- Actor loss (clipped surrogate) ---
        new_log_probs = np.log(probs[np.arange(m), actions] + 1e-8)
        ratio = np.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -np.mean(np.minimum(surr1, surr2))

        # --- Critic loss (MSE) ---
        critic_loss = np.mean((values - returns) ** 2)

        # --- Entropy bonus ---
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=-1))

        total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        # ---- Backprop ----
        # dL/d(values)
        d_values = 2.0 * value_coef * (values - returns) / m  # (batch,)

        # dL/d(logits)  — using REINFORCE-style gradient through softmax
        d_log_probs = np.zeros_like(probs)
        clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        mask = surr1 <= surr2  # where surr1 is active
        effective_adv = np.where(mask, ratio, np.sign(advantages) * clip_eps) * advantages / m  # (batch,)
        # Gradient through log_prob → prob: d(-log_prob * adv)/d(prob) = -adv/prob
        d_log_probs[np.arange(m), actions] = -effective_adv / (probs[np.arange(m), actions] + 1e-8)
        # Entropy gradient
        d_log_probs += entropy_coef * (np.log(probs + 1e-8) + 1.0) / m

        # Softmax Jacobian: dL/d(logits) = probs * (dL/d(log_probs) - sum)
        d_logits = probs * (d_log_probs - (d_log_probs * probs).sum(axis=-1, keepdims=True))

        # Gradients for actor head
        dW_actor = np.dot(a1.T, d_logits)
        db_actor = d_logits.sum(axis=0, keepdims=True)

        # Gradients for critic head
        d_values_col = d_values[:, np.newaxis]
        dW_critic = np.dot(a1.T, d_values_col)
        db_critic = d_values_col.sum(axis=0, keepdims=True)

        # Gradients through shared body
        d_a1 = np.dot(d_logits, self.W_actor.T) + np.dot(d_values_col, self.W_critic.T)
        d_z1 = d_a1.copy()
        d_z1[z1 <= 0] = 0  # ReLU derivative
        dW1 = np.dot(states.T, d_z1)
        db1 = d_z1.sum(axis=0, keepdims=True)

        # SGD update
        self.W1      -= self.lr * dW1
        self.b1      -= self.lr * db1
        self.W_actor -= self.lr * dW_actor
        self.b_actor -= self.lr * db_actor
        self.W_critic -= self.lr * dW_critic
        self.b_critic -= self.lr * db_critic

        return total_loss

    # ---- Serialization ----
    def get_weights(self):
        return {
            'W1': self.W1, 'b1': self.b1,
            'W_actor': self.W_actor, 'b_actor': self.b_actor,
            'W_critic': self.W_critic, 'b_critic': self.b_critic,
        }

    def set_weights(self, w):
        self.W1      = w['W1'];      self.b1      = w['b1']
        self.W_actor = w['W_actor']; self.b_actor = w['b_actor']
        self.W_critic= w['W_critic'];self.b_critic= w['b_critic']


# ---------------------------------------------------------------------------
# Rollout Buffer  (on-policy — cleared each update cycle)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores one episode's (or N-step) experience for on-policy PPO updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.log_probs   = []
        self.values      = []
        self.dones       = []

    def push(self, state, action, reward, log_prob, value, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """GAE (Generalised Advantage Estimation)."""
        rewards  = np.array(self.rewards,    dtype=np.float32)
        values   = np.array(self.values,     dtype=np.float32)
        dones    = np.array(self.dones,      dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_val = last_value
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae   = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            next_val = values[t]

        returns = advantages + values
        return returns, advantages

    def size(self):
        return len(self.states)

    def __len__(self):
        return len(self.states)


# ---------------------------------------------------------------------------
# PPO Agent  (drop-in replacement interface for DQNAgent)
# ---------------------------------------------------------------------------
class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Public interface mirrors DQNAgent so that MultiAgentController (or a PPO
    variant) can call  select_action / store_transition / learn / save / load
    with minimal changes.

    Key difference: PPO is ON-POLICY → no replay buffer.  `learn()` is called
    once per episode (or N steps), not every step.  The controller is
    responsible for calling `learn_episode()` at episode boundaries.
    """

    def __init__(
        self,
        state_dim:      int   = 5,
        action_dim:     int   = 2,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        lam:            float = 0.95,    # GAE lambda
        clip_eps:       float = 0.2,
        entropy_coef:   float = 0.01,
        value_coef:     float = 0.5,
        n_epochs:       int   = 4,
        batch_size:     int   = 64,
        epsilon:        float = 1.0,     # kept for API compatibility (unused by PPO)
        epsilon_decay:  float = 0.995,
        epsilon_min:    float = 0.01,
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.lam          = lam
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size

        # For API compatibility with DQNAgent callers
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.net    = ActorCriticMLP(state_dim, action_dim, lr=lr)
        self.buffer = RolloutBuffer()
        self.steps_done = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, training: bool = True, epsilon: float = None):
        """
        Sample from the policy distribution (training=True)
        or take argmax (training=False, greedy evaluation).
        Returns (action, log_prob, value) when training else (action,).
        The controller calls this; for PPO we always return a single int
        and store extras internally.
        """
        x = np.array([state])
        probs, value, _, _ = self.net.forward(x)
        probs = probs[0]; value = float(value[0])

        if training:
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = int(np.argmax(probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        # Store for buffer later (controller calls store_transition)
        self._last_log_prob = log_prob
        self._last_value    = value
        return int(action)

    # ------------------------------------------------------------------
    # Transition storage  (called by controller every step)
    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """Append one (s, a, r) tuple to the rollout buffer."""
        self.buffer.push(
            state    = state,
            action   = action,
            reward   = reward,
            log_prob = getattr(self, '_last_log_prob', 0.0),
            value    = getattr(self, '_last_value',    0.0),
            done     = done,
        )

    # ------------------------------------------------------------------
    # Per-step no-op (DQNAgent compatibility)
    # ------------------------------------------------------------------
    def learn(self):
        """Per-step call — PPO does nothing here; call learn_episode() instead."""
        return None

    # ------------------------------------------------------------------
    # Episode-level update
    # ------------------------------------------------------------------
    def learn_episode(self, last_value: float = 0.0):
        """
        Runs N epochs of mini-batch PPO updates over the current rollout buffer,
        then clears the buffer.
        """
        if len(self.buffer) < 2:
            self.buffer.clear()
            return 0.0

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=last_value, gamma=self.gamma, lam=self.lam
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states     = np.array(self.buffer.states,    dtype=np.float32)
        actions    = np.array(self.buffer.actions,   dtype=np.int32)
        old_lps    = np.array(self.buffer.log_probs, dtype=np.float32)
        returns    = returns.astype(np.float32)
        advantages = advantages.astype(np.float32)

        total_loss = 0.0
        n = len(states)
        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx  = indices[start: start + self.batch_size]
                if len(idx) < 2:
                    continue
                loss = self.net.update(
                    states[idx], actions[idx], returns[idx],
                    advantages[idx], old_lps[idx],
                    clip_eps=self.clip_eps,
                    entropy_coef=self.entropy_coef,
                    value_coef=self.value_coef,
                )
                total_loss += loss

        self.buffer.clear()
        self.steps_done += 1
        return total_loss

    # ------------------------------------------------------------------
    # Epsilon decay (API compatibility — not really used by PPO)
    # ------------------------------------------------------------------
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """No-op for PPO (no target network needed)."""
        pass

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.net.get_weights(), f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        self.net.set_weights(weights)
