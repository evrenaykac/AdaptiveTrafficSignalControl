import random
import numpy as np
import pickle
from collections import deque
import os

class SimpleMLP:
    def __init__(self, input_dim, output_dim, hidden_dim=64, learning_rate=1e-3, is_actor=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        self.is_actor = is_actor
        
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        if self.is_actor:
            # Gumbel Softmax continuous relaxation mimicking discrete probabilities 
            e = np.exp(self.z2 - np.max(self.z2, axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        return self.z2
    
    def backprop(self, X, grad_out):
        m = X.shape[0]
        dW2 = np.dot(self.a1.T, grad_out)
        db2 = np.sum(grad_out, axis=0, keepdims=True)
        
        grad_a1 = np.dot(grad_out, self.W2.T)
        grad_z1 = grad_a1.copy()
        grad_z1[self.z1 <= 0] = 0
        
        dW1 = np.dot(X.T, grad_z1)
        db1 = np.sum(grad_z1, axis=0, keepdims=True)
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train_mse(self, X, y):
        m = X.shape[0]
        y_pred = self.forward(X)
        grad_y_pred = (2.0 / m) * (y_pred - y)
        self.backprop(X, grad_y_pred)
        return np.mean((y_pred - y) ** 2)

    def get_weights(self): return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
    def set_weights(self, w): self.W1=w['W1']; self.b1=w['b1']; self.W2=w['W2']; self.b2=w['b2']

class DynamicReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, raw_metrics, next_state, done):
        # We store native metrics! No metric flushing needed when LLM weights change!
        self.buffer.append((state, action, raw_metrics, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, metrics, next_state, done = zip(*batch)
        return np.array(state), np.array(action), metrics, np.array(next_state), np.array(done)
    
    def __len__(self): return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim=5, action_dim=2, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, epsilon=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.actor = SimpleMLP(state_dim, action_dim, learning_rate=lr_actor, is_actor=True)
        self.target_actor = SimpleMLP(state_dim, action_dim, learning_rate=lr_actor, is_actor=True)
        
        # Critic Input: State + Action -> Q-value
        self.critic = SimpleMLP(state_dim + action_dim, 1, learning_rate=lr_critic)
        self.target_critic = SimpleMLP(state_dim + action_dim, 1, learning_rate=lr_critic)
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        self.memory = DynamicReplayBuffer(5000)
        self.batch_size = 64

    def select_action(self, state, training=True, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        if training and random.random() < eps:
            return random.randrange(self.action_dim)
            
        probs = self.actor.forward(np.array([state]))[0]
        return np.argmax(probs)

    def store_transition(self, state, action, raw_metrics, next_state, done):
        self.memory.push(state, action, raw_metrics, next_state, done)

    def learn(self, current_weights):
        if len(self.memory) < self.batch_size: return 0.0
        state, action, metrics_batch, next_state, done = self.memory.sample(self.batch_size)
        
        # Calculate dynamic target reward WITHOUT memory clear / flushing
        alpha = current_weights.get('alpha', 0.6)
        beta  = current_weights.get('beta',  0.3)
        gamma = current_weights.get('gamma', 0.1)
        
        rewards = np.array([-(alpha * m['queue'] + beta * m['wait'] + gamma * m['co2']) for m in metrics_batch]).reshape(-1, 1)
        done = done.reshape(-1, 1)

        # Build Action One-Hots
        action_one_hot = np.zeros((self.batch_size, self.action_dim))
        action_one_hot[np.arange(self.batch_size), action] = 1.0

        # Critic Update
        next_actions = self.target_actor.forward(next_state)
        target_q_next = self.target_critic.forward(np.hstack((next_state, next_actions)))
        target_q = rewards + (1 - done) * self.gamma * target_q_next
        
        critic_loss = self.critic.train_mse(np.hstack((state, action_one_hot)), target_q)
        
        # Actor Update
        pred_actions = self.actor.forward(state)
        # We need gradient of Critic w.r.t action. Simple implementation: Backprop through critic
        self.critic.forward(np.hstack((state, pred_actions)))
        m = self.batch_size
        grad_q = -1.0 / m * self.critic.W1[self.state_dim:, :] # Partial derivative
        
        actor_grad = np.dot(self.critic.a1, grad_q.T)
        self.actor.backprop(state, actor_grad)

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        return critic_loss

    def _soft_update(self, local, target):
        lw, tw = local.get_weights(), target.get_weights()
        for k in lw.keys():
            tw[k] = self.tau * lw[k] + (1 - self.tau) * tw[k]
        target.set_weights(tw)

    def update_epsilon(self): self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'actor': self.actor.get_weights(), 'critic': self.critic.get_weights()}, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                w = pickle.load(f)
                self.actor.set_weights(w['actor'])
                self.target_actor.set_weights(w['actor'])
                self.critic.set_weights(w['critic'])
                self.target_critic.set_weights(w['critic'])
