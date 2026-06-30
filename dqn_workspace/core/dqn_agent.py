import random
import numpy as np
import pickle
from collections import deque
import os

class SimpleMLP:
    def __init__(self, input_dim, output_dim, hidden_dim=64, learning_rate=1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # He Initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1) # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def train(self, X, y):
        m = X.shape[0]
        y_pred = self.forward(X)
        grad_y_pred = (2.0 / m) * (y_pred - y)
        
        dW2 = np.dot(self.a1.T, grad_y_pred)
        db2 = np.sum(grad_y_pred, axis=0, keepdims=True)
        
        grad_a1 = np.dot(grad_y_pred, self.W2.T)
        grad_z1 = grad_a1.copy()
        grad_z1[self.z1 <= 0] = 0
        
        dW1 = np.dot(X.T, grad_z1)
        db1 = np.sum(grad_z1, axis=0, keepdims=True)
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        return np.mean((y_pred - y) ** 2)

    def get_weights(self): return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
    def set_weights(self, w): self.W1=w['W1']; self.b1=w['b1']; self.W2=w['W2']; self.b2=w['b2']

# Dynamic Replay Buffer
class DynamicReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, raw_metrics, next_state, done):
        """
        raw_metrics: dict {'queue': float, 'wait': float, 'co2': float}
        We store the native metrics instead of a scalar reward. 
        When sampled, the reward is dynamically calculated using the LATEST LLM weights.
        """
        self.buffer.append((state, action, raw_metrics, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, metrics, next_state, done = zip(*batch)
        return np.array(state), np.array(action), metrics, np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim=5, action_dim=2, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.policy_net = SimpleMLP(state_dim, action_dim, learning_rate=lr)
        self.target_net = SimpleMLP(state_dim, action_dim, learning_rate=lr)
        self.update_target_network()
        
        self.memory = DynamicReplayBuffer(5000)
        self.batch_size = 32
        self.steps_done = 0

    def select_action(self, state, training=True, epsilon=None):
        eps = epsilon if epsilon is not None else self.epsilon
        if training and random.random() < eps:
            return random.randrange(self.action_dim)
        
        q_values = self.policy_net.forward(np.array([state]))
        return np.argmax(q_values[0])

    def store_transition(self, state, action, raw_metrics, next_state, done):
        self.memory.push(state, action, raw_metrics, next_state, done)

    def learn(self, current_weights):
        """
        current_weights: dynamic reward weights {'alpha': float, 'beta': float, 'gamma': float}
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        state, action, metrics_batch, next_state, done = self.memory.sample(self.batch_size)
        
        # Calculate rewards dynamically based on latest LLM weights
        alpha = current_weights.get('alpha', 0.6)
        beta  = current_weights.get('beta',  0.3)
        gamma = current_weights.get('gamma', 0.1)
        
        rewards = []
        for m in metrics_batch:
            # We want to MINIMIZE the negative costs, equivalent to max reward
            r = -(alpha * m['queue'] + beta * m['wait'] + gamma * m['co2'])
            rewards.append(r)
        reward_arr = np.array(rewards)
        
        current_q = self.policy_net.forward(state)
        next_q_target = self.target_net.forward(next_state)
        target_q = current_q.copy()
        
        batch_index = np.arange(self.batch_size)
        max_next_q = np.max(next_q_target, axis=1)
        target_q[batch_index, action] = reward_arr + self.gamma * max_next_q * (1 - done)
        
        loss = self.policy_net.train(state, target_q)
        self.steps_done += 1
        return loss
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.policy_net.get_weights(), f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                weights = pickle.load(f)
                self.policy_net.set_weights(weights)
                self.target_net.set_weights(weights)
