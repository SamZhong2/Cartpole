# run.py
import torch
import time
from dqn import DQN


def load_and_run_model(env, model_path, episodes=5):
    """Load a pre-trained model and run it in the environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for e in range(episodes):
        state, _ = env.reset()  # Update to properly extract the initial state
        for time_step in range(500):
            env.render()  # Render the environment
            time.sleep(0.01)  # Slow down the rendering for visibility
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = model(state)
            action = torch.argmax(action_values, dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            if done:
                print(f"Episode: {e + 1}/{episodes}, Score: {time_step}")
                break
    env.close()
    print("Game finished.\n")
