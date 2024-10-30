import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Initial agent position 
beehive_position = np.array([0.0, 0.0])
agent_position = beehive_position.copy()
path_x = [agent_position[0]]
path_y = [agent_position[1]]

# Global variables
success_flag = False  # Track if the agent has successfully found the reward
movement_vector = np.array([0.0, 0.0])  # Compass for tracking movement

# Randomize reward position
def randomize_reward_position():
    return np.array([random.uniform(-10, 10), random.uniform(-10, 10)])

reward_position = randomize_reward_position()  # Initial reward location

# Vision parameters
vision_distance = 4.0  # Maximum distance the agent can see
vision_angle = 100.0  # Field of view angle in degrees
gaze_direction = 0.0  # Initial direction the agent is facing (0 degrees)

# Set up the dot, path plot, and vision line
agent_dot, = ax.plot([], [], 'bo', markersize=8)
path_trace, = ax.plot([], [], 'r-', lw=1)
reward_dot, = ax.plot([reward_position[0]], [reward_position[1]], 'go', markersize=10, label="Reward")
vision_line, = ax.plot([], [], 'g--', lw=0.8, label="Vision")

# Update the vision cone to be dynamic
vision_cone = patches.Wedge((0, 0), vision_distance, 0, 0, color='green', alpha=0.2)
ax.add_patch(vision_cone)

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # Input layer: consistent state size of 4
        self.fc2 = nn.Linear(64, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 8)   # Output layer: Q-values for 8 possible actions

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# Initialize the Q-Network and optimizer
q_network = QNetwork()
optimizer = optim.Adam(q_network.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Define possible actions (movements)
actions = np.array([
    [-0.1, 0], [0.1, 0], [0, -0.1], [0, 0.1],  # left, right, down, up
    [-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]  # diagonals
])

def get_state():
    """Returns the state as a vector of agent's position and vision information if the reward is visible."""
    vector_to_reward = reward_position - agent_position
    if is_within_vision():
        return np.concatenate((agent_position, vector_to_reward))
    else:
        # If the reward is not visible, return the agent's position and a zero vector for the reward
        return np.concatenate((agent_position, [0.0, 0.0]))


def is_within_vision():
    """Checks if the reward is within the agent's vision range."""
    vector_to_reward = reward_position - agent_position
    distance_to_reward = np.linalg.norm(vector_to_reward)
    if distance_to_reward > vision_distance:
        return False

    # Calculate angle between the agent's facing direction and the vector to the reward
    agent_direction = np.array([np.cos(np.radians(gaze_direction)), np.sin(np.radians(gaze_direction))])
    unit_vector_to_reward = vector_to_reward / distance_to_reward
    dot_product = np.dot(agent_direction, unit_vector_to_reward)
    angle_to_reward = np.degrees(np.arccos(dot_product))

    return angle_to_reward <= vision_angle / 2.0

def choose_action(state, epsilon):
    """Chooses an action based on an epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randint(0, len(actions) - 1)
    else:
        state_tensor = torch.FloatTensor(state)
        q_values = q_network(state_tensor)
        return torch.argmax(q_values).item()

def update_q_network(state, action, reward, next_state):
    """Updates the Q-network based on the reward received and the next state."""
    state_tensor = torch.FloatTensor(state)
    next_state_tensor = torch.FloatTensor(next_state)

    # Get predicted Q-values for current and next state
    q_values = q_network(state_tensor)
    next_q_values = q_network(next_state_tensor)

    # Calculate target Q-value
    target_q_value = reward + 0.99 * torch.max(next_q_values).item()
    target = q_values.clone()
    target[action] = target_q_value

    # Optimize the Q-network
    optimizer.zero_grad()
    loss = criterion(q_values, target)
    loss.backward()
    optimizer.step()

def get_reward(old_position, new_position):
    """Returns a positive reward if the agent moves closer to the reward or a negative one if moving towards the boundary."""
    old_distance = np.linalg.norm(reward_position - old_position)
    new_distance = np.linalg.norm(reward_position - new_position)

    if np.any(new_position < -10) or np.any(new_position > 10):
        return -1.0  # Punishment for moving out of bounds
    return 1.0 if new_distance < old_distance else -1.0

def keep_within_bounds(position):
    """Adjusts the agent's position to ensure it stays within the boundary."""
    position[0] = np.clip(position[0], -10, 10)
    position[1] = np.clip(position[1], -10, 10)
    return position

def init():
    """Initializes the animation."""
    agent_dot.set_data([], [])
    path_trace.set_data([], [])
    reward_dot.set_data([reward_position[0]], [reward_position[1]])
    vision_line.set_data([], [])
    vision_cone.set_center((0, 0))  # Make it dynamic by setting its center properly
    return agent_dot, path_trace, reward_dot, vision_line, vision_cone

def check_success(agent_position, reward_position, threshold=0.5):
    """Returns True if the agent reaches the reward within a certain threshold."""
    distance_to_reward = np.linalg.norm(reward_position - agent_position)
    return distance_to_reward < threshold

def animate(i):
    """Animates the agent movement."""
    global agent_position, success_flag, movement_vector, reward_position, gaze_direction

    if success_flag:
        # Update reward and reset the agent to the hive after success
        reward_position = randomize_reward_position()
        agent_position = beehive_position.copy()
        movement_vector = np.array([0.0, 0.0])
        success_flag = False  # Reset success flag
        print("New Reward Position:", reward_position)

    state = get_state()
    action_index = choose_action(state, epsilon=0.1)
    action = actions[action_index]

    old_position = agent_position.copy()
    agent_position += action
    agent_position = keep_within_bounds(agent_position)  # Keep agent within the boundary

    movement_vector += action  # Update the movement vector

    next_state = get_state()
    reward = get_reward(old_position, agent_position)
    update_q_network(state, action_index, reward, next_state)

    path_x.append(agent_position[0])
    path_y.append(agent_position[1])

    if check_success(agent_position, reward_position):
        print("Pollen collected! Returning to hive.")
        success_flag = True
        return agent_dot, path_trace, reward_dot, vision_line, vision_cone

    # Update agent's dot, path trace, and vision line
    agent_dot.set_data([agent_position[0]], [agent_position[1]])
    path_trace.set_data(path_x, path_y)
    reward_dot.set_data([reward_position[0]], [reward_position[1]])

    # Update gaze direction randomly to simulate looking around
    gaze_direction += random.uniform(-20.0, 20.0)
    gaze_direction %= 360  # Ensure the angle stays within [0, 360] degrees

    # Update the vision cone based on the agent's position and orientation
    vision_cone.set_center(agent_position)
    vision_cone.set_theta1(gaze_direction - vision_angle / 2)
    vision_cone.set_theta2(gaze_direction + vision_angle / 2)

    return agent_dot, path_trace, reward_dot, vision_line, vision_cone

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=10, blit=True)

plt.legend()
plt.show()
