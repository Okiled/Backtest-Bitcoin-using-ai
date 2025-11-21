import numpy as np
import torch
from models.lstm_model import LSTMConfig, LSTMTrainer

def test_lstm_forward_pass():
    config = LSTMConfig(look_back=5, hidden_size=4, num_layers=1, dropout=0.0, learning_rate=0.01, batch_size=4, epochs=1, patience=1, weight_decay=0.0)
    trainer = LSTMTrainer(config, torch.device("cpu"))
    features = np.random.rand(20, 3)
    targets = np.random.rand(20)
    trainer.fit(features, features, targets, targets)
    preds = trainer.predict(features)
    assert preds.shape[0] == len(features) - config.look_back

def test_simple_ppo_predict():
    from models.ppo_agent import PPOConfig, SimplePPOAgent
    agent = SimplePPOAgent(PPOConfig(learning_rate=0.001, gamma=0.99, n_steps=10, ent_coef=0.0, vf_coef=0.5, clip_range=0.2, batch_size=4, total_timesteps=10))
    action, _ = agent.predict(np.array([0.1, 0.2]))
    assert action in (0, 1)
