import os
import neat
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pickle
from agent import Agent, distance
from constants import *
import random

def load_genome(filepath):
    """Load a genome from a pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['genome'], data['config'], data['generation']

class NetworkVisualizer:
    def __init__(self, genome, config):
        """Initialize network visualization."""
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Create node positions dictionary
        self.node_positions = {}
        
        # Input nodes (including bias)
        num_inputs = len(config.genome_config.input_keys)
        for i, key in enumerate(config.genome_config.input_keys):
            self.node_positions[key] = (0.1, 0.9 - (i * 0.8 / (num_inputs-1)))
            
        # Output nodes
        num_outputs = len(config.genome_config.output_keys)
        for i, key in enumerate(config.genome_config.output_keys):
            self.node_positions[key] = (0.9, 0.9 - (i * 0.8 / (num_outputs-1)))
            
        # Hidden nodes - position them in layers
        hidden_nodes = [n for n in genome.nodes.keys() 
                       if n not in config.genome_config.input_keys and 
                       n not in config.genome_config.output_keys]
                       
        if hidden_nodes:
            for i, node in enumerate(hidden_nodes):
                layer = 0.5  # All hidden nodes in middle for simplicity
                self.node_positions[node] = (layer, 0.9 - (i * 0.8 / (len(hidden_nodes))))
        
        # Create color map for activation visualization
        self.cmap = LinearSegmentedColormap.from_list('activation', ['blue', 'gray', 'red'])
        
        # Store node activations
        self.node_activations = {node_id: 0.0 for node_id in self.node_positions.keys()}

    def update_activations(self, inputs):
        """Update node activations based on new inputs."""
        # Get individual node activations by accessing the network internals
        self.node_activations = {node_id: 0.0 for node_id in self.node_positions.keys()}
        
        # Set input node activations
        for i, val in enumerate(inputs):
            if i in self.node_positions:
                self.node_activations[i] = val
        
        # Get network output and hidden node activations
        outputs = self.net.activate(inputs)
        
        # Set output node activations
        output_keys = self.config.genome_config.output_keys
        for i, val in enumerate(outputs):
            self.node_activations[output_keys[i]] = val
            
        # Hidden node activations are trickier - we approximate them based on connection weights
        # This is a simplified approach
        for node_id in self.node_positions.keys():
            if node_id not in self.config.genome_config.input_keys and \
               node_id not in self.config.genome_config.output_keys:
                incoming = []
                for conn in self.genome.connections.values():
                    if conn.enabled and conn.key[1] == node_id:
                        in_val = self.node_activations[conn.key[0]]
                        incoming.append(in_val * conn.weight)
                if incoming:
                    self.node_activations[node_id] = np.tanh(sum(incoming))
        
        return outputs

    def draw(self, ax):
        """Draw the network with current activation states."""
        ax.clear()
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        # Draw connections
        for conn in self.genome.connections.values():
            if conn.enabled and conn.key[0] in self.node_positions and conn.key[1] in self.node_positions:
                start = self.node_positions[conn.key[0]]
                end = self.node_positions[conn.key[1]]
                weight = conn.weight
                color = 'red' if weight > 0 else 'blue'
                alpha = min(abs(weight), 1.0)
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=color, alpha=alpha, linewidth=1)
        
        # Draw nodes with activation colors
        for node_id, pos in self.node_positions.items():
            activation = self.node_activations[node_id]
            color = self.cmap((activation + 1) / 2)  # Map from [-1,1] to [0,1]
            
            circle = plt.Circle(pos, 0.02, color=color)
            ax.add_patch(circle)
            
            # Add node label and activation value
            label = f"{node_id}\n{activation:.2f}"
            ax.annotate(label, pos, xytext=(0, -10), textcoords='offset points',
                       ha='center', va='top', fontsize=8)
        
        ax.set_title("Neural Network Activation States")
        ax.axis('off')

def run_simulation(genome_path):
    """Run continuous simulation with network visualization."""
    # Load genome
    genome, config, generation = load_genome(genome_path)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121)  # Simulation
    ax2 = fig.add_subplot(122)  # Network visualization
    
    # Initialize network visualizer
    net_viz = NetworkVisualizer(genome, config)
    
    # Initialize simulation state
    state = {
        'agent': Agent((WIDTH/2, HEIGHT/2)),
        'reward_pos': None,
        'frame': 0
    }
    
    def reset_reward():
        """Generate new reward position."""
        while True:
            pos = (random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE),
                  random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE))
            dist_to_home = distance((WIDTH/2, HEIGHT/2), pos)
            if 50 <= dist_to_home <= 150:
                return pos
    
    state['reward_pos'] = reset_reward()
    
    def update(frame):
        state['frame'] = frame
        
        # Clear simulation plot
        ax1.clear()
        ax1.set_xlim(0, WIDTH)
        ax1.set_ylim(0, HEIGHT)
        
        # Get neural network input and update agent
        inputs = state['agent'].get_sensor_data(state['reward_pos'])
        outputs = net_viz.update_activations(inputs)
        done = state['agent'].update(outputs, state['reward_pos'])
        
        # Draw agent path
        if len(state['agent'].path_positions) > 1:
            path_x = [p[0] for p in state['agent'].path_positions[-50:]]  # Show last 50 positions
            path_y = [p[1] for p in state['agent'].path_positions[-50:]]
            ax1.plot(path_x, path_y, 'b-', alpha=0.5)
        
        # Draw agent
        agent_circle = patches.Circle(
            (state['agent'].x, state['agent'].y),
            AGENT_SIZE,
            color='red' if state['agent'].carrying_reward else 'green'
        )
        ax1.add_patch(agent_circle)
        
        # Draw direction indicator
        end_x = state['agent'].x + AGENT_SIZE * 2 * math.cos(math.radians(state['agent'].angle))
        end_y = state['agent'].y + AGENT_SIZE * 2 * math.sin(math.radians(state['agent'].angle))
        ax1.plot([state['agent'].x, end_x], [state['agent'].y, end_y], 'k-', linewidth=2)
        
        # Draw beehive
        beehive_rect = patches.Rectangle(
            (WIDTH/2 - BEEHIVE_SIZE/2, HEIGHT/2 - BEEHIVE_SIZE/2),
            BEEHIVE_SIZE, BEEHIVE_SIZE, color='brown'
        )
        ax1.add_patch(beehive_rect)
        
        # Draw reward
        reward_circle = patches.Circle(
            state['reward_pos'],
            REWARD_SIZE,
            color='red'
        )
        ax1.add_patch(reward_circle)
        
        # Add status text
        status_text = [
            f"Steps: {state['agent'].steps_taken}",
            f"Energy: {state['agent'].energy:.0f}",
            f"Carrying: {'YES' if state['agent'].carrying_reward else 'NO'}"
        ]
        for i, text in enumerate(status_text):
            ax1.text(5, HEIGHT - 5 - (i * 15), text,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # Reset if task completed or failed
        if done or state['agent'].returned_home:
            if state['agent'].returned_home:
                print(f"Successfully returned home after {state['agent'].steps_taken} steps!")
            state['agent'] = Agent((WIDTH/2, HEIGHT/2))
            state['reward_pos'] = reset_reward()
        
        # Update network visualization
        net_viz.draw(ax2)
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    # Specify the path to your genome file
    genome_path = "runs/run_20241122_215413/genomes/champion_genome_gen825.pkl"  # Update this path
    run_simulation(genome_path)