"""
A2C Agent for Multi-Agent Traffic Signal Control.
Uses a NumPy-based Actor-Critic network to match the existing DQN/PPO stack.
Interface is intentionally compatible with DQNAgent so MultiAgentController can be reused.
"""

import numpy as np
import pickle

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

    # ---- Update (SGD for pure A2C) ----
    def update(self,
               states: np.ndarray,
               actions: np.ndarray,
               returns: np.ndarray,
               advantages: np.ndarray,
               entropy_coef: float = 0.01,
               value_coef: float = 0.5):
        """One gradient step of standard A2C loss."""
        probs, values, a1, z1 = self.forward(states)
        m = states.shape[0]

        # --- Actor loss ---
        # A2C Policy Gradient Loss: -1 * log(pi(a|s)) * Advantage
        # Equivalent to cross-entropy loss weighted by advantages
        
        # --- Critic loss (MSE) ---
        critic_loss = np.mean((values - returns) ** 2)

        # --- Entropy bonus ---
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=-1))
        
        # dL/d(values) -> gradient of Critic MSE loss w.r.t values
        d_values = 2.0 * value_coef * (values - returns) / m  # (batch,)

        # dL/d(logits) -> gradient of Actor loss w.r.t logits
        # For policy gradient: d(-log(p)*A) / d(logits) = A * (p - one_hot(action))
        d_logits = np.zeros_like(probs)
        for i in range(m):
            # p_i
            p = probs[i]
            # one hot
            one_hot = np.zeros_like(p)
            one_hot[actions[i]] = 1.0
            
            # Policy gradient
            pg_grad = advantages[i] * (p - one_hot)
            # Entropy gradient
            ent_grad = entropy_coef * (np.log(p + 1e-8) + 1.0)
            
            # Combine: pg_grad subtracts entropy (maximize obj = minimize -obj)
            # Standard: minimize Policy Loss - Entropy Bonus
            # PGLoss = -E[log_pi * Adv], grad = -Adv * grad_log_pi = -Adv * (1/pi) * pi * (1-pi) ...
            # Actually, standard derivative of -log_pi * A w.r.t logits is A * (probs - 1)
            # which matches `advantages[i] * (p - one_hot)`
            d_logits[i] = pg_grad / m - (p * (ent_grad - np.sum(p * ent_grad))) / m

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
        self.W1       -= self.lr * dW1
        self.b1       -= self.lr * db1
        self.W_actor  -= self.lr * dW_actor
        self.b_actor  -= self.lr * db_actor
        self.W_critic -= self.lr * dW_critic
        self.b_critic -= self.lr * db_critic

        # Calculate pure actor loss for reporting
        actor_loss = -np.mean(np.log(probs[np.arange(m), actions] + 1e-8) * advantages)
        total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
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
# Rollout Buffer (on-policy — cleared each update cycle)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores one episode's experience for on-policy A2C updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.values      = []
        self.dones       = []

    def push(self, state, action, reward, value, done=False):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
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
# A2C Agent
# ---------------------------------------------------------------------------
class A2CAgent:
    """
    Advantage Actor-Critic agent.
    Public interface mirrors DQNAgent/PPOAgent.
    """

    def __init__(
        self,
        state_dim:      int   = 5,
        action_dim:     int   = 2,
        lr:             float = 3e-4,
        gamma:          float = 0.99,
        lam:            float = 0.95,    # GAE lambda
        entropy_coef:   float = 0.01,
        value_coef:     float = 0.5,
        epsilon:        float = 1.0,     # kept for API compatibility 
        epsilon_decay:  float = 0.995,
        epsilon_min:    float = 0.01,
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.lam          = lam
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef

        # API compatibility
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
        x = np.array([state])
        probs, value, _, _ = self.net.forward(x)
        probs = probs[0]; value = float(value[0])

        if training:
            action = np.random.choice(self.action_dim, p=probs)
        else:
            action = int(np.argmax(probs))

        # Store value for buffer later
        self._last_value = value
        return int(action)

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(
            state    = state,
            action   = action,
            reward   = reward,
            value    = getattr(self, '_last_value', 0.0),
            done     = done,
        )

    # ------------------------------------------------------------------
    # Per-step no-op (DQNAgent compatibility)
    # ------------------------------------------------------------------
    def learn(self):
        return None

    # ------------------------------------------------------------------
    # Episode-level update
    # ------------------------------------------------------------------
    def learn_episode(self, last_value: float = 0.0):
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
        returns    = returns.astype(np.float32)
        advantages = advantages.astype(np.float32)

        # A2C typically updates over the entire batch at once (or minibatches)
        # We will update once over the entire rollout buffer here.
        loss = self.net.update(
            states, actions, returns,
            advantages,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
        )

        self.buffer.clear()
        self.steps_done += 1
        return loss

    # ------------------------------------------------------------------
    # Epsilon decay & target network (API compatibility)
    # ------------------------------------------------------------------
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
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
