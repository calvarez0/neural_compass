import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Initial agent position
agent_position = np.array([0.0, 0.0])
path_x = [agent_position[0]]
path_y = [agent_position[1]]

# Define the neural network structure
input_neurons = 2  # X and Y coordinates as inputs
hidden_neurons = 4  # Intermediate processing neurons
output_neurons = 2  # Output for new X and Y positions

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.uniform(-1, 1, (hidden_neurons, input_neurons))
bias_hidden = np.random.uniform(-1, 1, hidden_neurons)
weights_hidden_output = np.random.uniform(-1, 1, (output_neurons, hidden_neurons))
bias_output = np.random.uniform(-1, 1, output_neurons)

# Set up the dot and path plot
agent_dot, = ax.plot([], [], 'bo', markersize=8)
path_trace, = ax.plot([], [], 'r-', lw=1)

def activation_function(x):
    """Applies the tanh activation function."""
    return np.tanh(x)

def neural_network(input_vector):
    """Processes the input vector through the neural network."""
    # Hidden layer computation
    hidden_layer_input = np.dot(weights_input_hidden, input_vector) + bias_hidden
    hidden_layer_output = activation_function(hidden_layer_input)

    # Output layer computation
    output_layer_input = np.dot(weights_hidden_output, hidden_layer_output) + bias_output
    output = activation_function(output_layer_input)

    # Scale output to control movement magnitude
    return output * 0.1

def init():
    """Initializes the animation."""
    agent_dot.set_data([], [])
    path_trace.set_data([], [])
    return agent_dot, path_trace

def update_position():
    """Update agent position based on neural network output."""
    global agent_position
    input_vector = agent_position  # Current position as input
    movement = neural_network(input_vector)
    agent_position += movement

def animate(i):
    """Animates the agent movement."""
    update_position()
    path_x.append(agent_position[0])
    path_y.append(agent_position[1])

    # Log the agent position for debugging
    print(f"Frame {i}: Agent Position - {agent_position}")

    # Set the data for the dot and path
    agent_dot.set_data([agent_position[0]], [agent_position[1]])  # Make x and y sequences
    path_trace.set_data(path_x, path_y)

    return agent_dot, path_trace

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100, blit=True)

plt.show()
