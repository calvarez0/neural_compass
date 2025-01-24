import os
import pickle
import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from neat.nn import FeedForwardNetwork
import math
import random
from agent import Agent
from constants import *
from agent import *


def get_node_activations(network, genome, inputs, hidden_nodes):
    """
    Get hidden node activations using the same approach as NetworkVisualizer.
    """
    # Initialize activations dictionary
    activations = {node_id: 0.0 for node_id in hidden_nodes}
    
    # First get network outputs normally
    outputs = network.activate(inputs)
    
    # Now compute hidden node activations based on incoming connections
    for node_id in hidden_nodes:
        incoming = []
        for conn in genome.connections.values():
            if conn.enabled and conn.key[1] == node_id:
                # For input nodes, we can get values directly from inputs
                if conn.key[0] < 0:  # Input nodes have negative IDs
                    in_val = inputs[abs(conn.key[0]) - 1]  # -1 because input IDs start at -1
                    incoming.append(in_val * conn.weight)
                else:
                    # For other nodes, we need to recursively compute their values
                    # For simplicity and to avoid cycles, we'll just use 0 for now
                    incoming.append(0)
        
        if incoming:
            # Apply tanh activation like in your NetworkVisualizer
            activations[node_id] = np.tanh(sum(incoming))
    
    return activations


def analyze_neural_compass(genome_path):
    """
    Analyze a NEAT genome for compass-like neural activation patterns using FFT analysis.
    """
    # Load the genome
    print(f"Loading genome from {genome_path}")
    with open(genome_path, 'rb') as f:
        data = pickle.load(f)
        genome = data['genome']
        config = data['config']
    
    # Create the neural network
    network = FeedForwardNetwork.create(genome, config)
    
    # Get all node IDs
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [n for n in genome.nodes.keys() 
                   if n not in input_nodes and n not in output_nodes]
    
    print(f"\nNetwork structure:")
    print(f"Input nodes: {input_nodes}")
    print(f"Hidden nodes: {hidden_nodes}")
    print(f"Output nodes: {output_nodes}")
    
    # Collect activation patterns for all angles
    angles = np.linspace(0, 360, 360, endpoint=False)
    node_activations = {node: np.zeros(len(angles)) for node in hidden_nodes}
    
    print("\nCollecting activation patterns...")
    
    # For each angle, collect activations
    for i, angle in enumerate(angles):
        # Create input vector - we only vary the heading angle
        heading_x = np.cos(np.radians(angle))
        heading_y = np.sin(np.radians(angle))
        
        # Create full input vector (matching your agent's structure)
        inputs = [
            heading_x, heading_y,  # Heading vector
            0, 0,                 # Path integration (zeroed)
            0, 0,                 # Target vector (zeroed)
            0, 0,                 # State flags
            0, 0, 0, 0, 0, 0, 0, 0, 0  # Memory grid (zeroed)
        ]
        
        # Get hidden node activations
        activations = get_node_activations(network, genome, inputs, hidden_nodes)
        
        # Store activations
        for node in hidden_nodes:
            node_activations[node][i] = activations[node]
    
    def analyze_sinusoidal_pattern(signal):
        """
        Analyze a signal for sinusoidal behavior using FFT.
        """
        if len(signal) == 0 or np.all(signal == 0):
            return 0, 0
        
        # Calculate FFT
        fft_vals = np.abs(fft(signal))
        freqs = fftfreq(len(signal))
        
        # Get positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        # Find dominant frequency
        dominant_idx = np.argmax(fft_vals)
        dominant_freq = freqs[dominant_idx]
        
        # Calculate period
        period = abs(1 / (dominant_freq * len(signal)) * 360) if dominant_freq != 0 else 0
        
        # Calculate purity (ratio of dominant frequency power to total power)
        total_power = np.sum(fft_vals)
        purity = fft_vals[dominant_idx] / total_power if total_power > 0 else 0
        
        return period, purity
    
    # Find compass neurons
    compass_neurons = []
    print("\nAnalyzing activation patterns for compass-like behavior...")
    
    for node in hidden_nodes:
        activations = node_activations[node]
        if np.any(activations != 0):
            period, purity = analyze_sinusoidal_pattern(activations)
            
            # Criteria for compass neurons:
            # - Period should be close to 360° (one full rotation)
            # - Signal should be mostly sinusoidal (high purity)
            if 330 <= period <= 390 and purity > 0.7:  # Slightly relaxed purity threshold
                compass_neurons.append({
                    'node_id': node,
                    'period': period,
                    'purity': purity,
                    'activations': activations
                })
    
    print(f"\nFound {len(compass_neurons)} potential compass neurons:")
    for neuron in compass_neurons:
        print(f"Node {neuron['node_id']}: Period={neuron['period']:.1f}°, Purity={neuron['purity']:.2f}")
    
    # Visualize results
    if compass_neurons:
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Linear activation over rotation
        ax1 = plt.subplot(211)
        for neuron in compass_neurons:
            ax1.plot(angles, neuron['activations'], 
                    label=f"Node {neuron['node_id']}")
        
        ax1.set_title("Compass Neuron Activation Patterns")
        ax1.set_xlabel("Rotation Angle (degrees)")
        ax1.set_ylabel("Activation")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Polar plot showing directional nature
        ax2 = plt.subplot(212, projection='polar')
        for neuron in compass_neurons:
            ax2.plot(np.radians(angles), neuron['activations'])
            
        ax2.set_title("Polar View of Activation Patterns")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def analyze_sinusoidal_pattern(signal):
    """
    Analyze a signal for sinusoidal behavior using FFT.
    Returns period and purity score.
    """
    if len(signal) == 0 or np.all(signal == 0):
        return 0, 0
    
    # Calculate FFT
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(len(signal))
    
    # Get positive frequencies only
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]
    
    # Find dominant frequency
    dominant_idx = np.argmax(fft_vals)
    dominant_freq = freqs[dominant_idx]
    
    # Calculate period
    period = abs(1 / (dominant_freq * len(signal)) * 360) if dominant_freq != 0 else 0
    
    # Calculate purity (ratio of dominant frequency power to total power)
    total_power = np.sum(fft_vals)
    purity = fft_vals[dominant_idx] / total_power if total_power > 0 else 0
    
    return period, purity


def visualize_agent_and_compass(genome_path):
    """
    Create a visualization showing both the agent's behavior and the compass neuron activation.
    Combines the agent simulation from activation_test.py with compass neuron visualization.
    """
    # First analyze to find the compass neuron
    print("Finding compass neurons...")
    with open(genome_path, 'rb') as f:
        data = pickle.load(f)
        genome = data['genome']
        config = data['config']
    
    network = FeedForwardNetwork.create(genome, config)
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [n for n in genome.nodes.keys() 
                   if n not in input_nodes and n not in output_nodes]
    
    # Find the best compass neuron first
    angles = np.linspace(0, 360, 360, endpoint=False)
    node_activations = {node: np.zeros(len(angles)) for node in hidden_nodes}
    
    # Collect activation data for all neurons
    for i, angle in enumerate(angles):
        heading_x = np.cos(np.radians(angle))
        heading_y = np.sin(np.radians(angle))
        inputs = [heading_x, heading_y] + [0] * 15  # Rest of inputs zeroed
        activations = get_node_activations(network, genome, inputs, hidden_nodes)
        for node in hidden_nodes:
            node_activations[node][i] = activations[node]
    
    # Find best compass neuron
    best_neuron = None
    best_purity = 0
    for node in hidden_nodes:
        activations = node_activations[node]
        if np.any(activations != 0):
            period, purity = analyze_sinusoidal_pattern(activations)
            if 330 <= period <= 390 and purity > best_purity:
                best_neuron = node
                best_purity = purity
                best_activation_pattern = activations
    
    if not best_neuron:
        print("No compass neurons found!")
        return
        
    print(f"Found compass neuron {best_neuron} with purity {best_purity:.2f}")
    
    def reset_reward():
        """Generate new reward position."""
        while True:
            pos = (random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE),
                  random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE))
            dist_to_home = distance((WIDTH/2, HEIGHT/2), pos)
            if 50 <= dist_to_home <= 150:
                return pos

    # Now set up the visualization
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, width_ratios=[2, 1, 1], figure=fig)
    
    # Agent simulation plot
    ax_sim = fig.add_subplot(gs[0])
    ax_sim.set_xlim(0, WIDTH)
    ax_sim.set_ylim(0, HEIGHT)
    ax_sim.set_title("Agent Behavior")
    ax_sim.set_aspect('equal')  # Add this to prevent squishing
    
    # Compass activation plot
    ax_activation = fig.add_subplot(gs[1])
    ax_activation.set_title(f"Compass Neuron {best_neuron}")
    ax_activation.set_ylim(-1.2, 1.2)
    ax_activation.set_xlim(0, 360)
    ax_activation.grid(True)
    
    # Polar plot
    ax_compass = fig.add_subplot(gs[2], projection='polar')
    ax_compass.set_title("Direction")
    
    # Initialize state
    state = {
        'agent': Agent((WIDTH/2, HEIGHT/2)),
        'reward_pos': reset_reward(),
        'frame': 0,
        'path_line': None,
        'path': []  # Add this to store path points
    }
    
    # Create visualization elements
    agent_circle = patches.Circle((state['agent'].x, state['agent'].y), AGENT_SIZE, color='green')
    ax_sim.add_patch(agent_circle)
    
    direction_line, = ax_sim.plot([0, 0], [0, 0], 'k-', linewidth=2)
    
    beehive = patches.Rectangle(
        (WIDTH/2 - BEEHIVE_SIZE/2, HEIGHT/2 - BEEHIVE_SIZE/2),
        BEEHIVE_SIZE, BEEHIVE_SIZE, color='brown'
    )
    ax_sim.add_patch(beehive)
    
    reward = patches.Circle(state['reward_pos'], REWARD_SIZE, color='red')
    ax_sim.add_patch(reward)
    
    # Set up activation plot
    activation_line, = ax_activation.plot(best_activation_pattern, 'b-')
    current_point, = ax_activation.plot([0], [0], 'ro', markersize=10)
    
    # Set up compass plot
    compass_marker, = ax_compass.plot([], [], 'go', markersize=10)
    compass_line, = ax_compass.plot([], [], 'k-', linewidth=2)
    
    # Text for angle and activation
    info_text = ax_sim.text(5, HEIGHT-5, '', verticalalignment='top',
                           bbox=dict(facecolor='white', alpha=0.7))
    
    # Initialize path line
    path_line, = ax_sim.plot([], [], 'b-', alpha=0.5)
    
    def init():
        agent_circle.center = (WIDTH/2, HEIGHT/2)
        direction_line.set_data([], [])
        compass_marker.set_data([], [])
        compass_line.set_data([], [])
        current_point.set_data([], [])
        path_line.set_data([], [])
        info_text.set_text('')
        state['path'] = []
        reward.center = state['reward_pos']  # Add this line to properly initialize reward position
        return agent_circle, direction_line, compass_marker, compass_line, current_point, info_text, path_line, reward  # Make sure reward is in return tuple

    def update(frame):
        state['frame'] = frame
        
        # Update agent
        inputs = state['agent'].get_sensor_data(state['reward_pos'])
        outputs = network.activate(inputs)
        done = state['agent'].update(outputs, state['reward_pos'])
        
        # Update path
        state['path'].append((state['agent'].x, state['agent'].y))
        if len(state['path']) > 150:
            state['path'] = state['path'][-150:]
        if len(state['path']) > 1:
            path_x, path_y = zip(*state['path'])
            path_line.set_data(path_x, path_y)
        
        # Update agent visualization
        agent_circle.center = (state['agent'].x, state['agent'].y)
        agent_circle.set_facecolor('red' if state['agent'].carrying_reward else 'green')
        
        # Update direction indicator
        end_x = state['agent'].x + AGENT_SIZE * 2 * math.cos(math.radians(state['agent'].angle))
        end_y = state['agent'].y + AGENT_SIZE * 2 * math.sin(math.radians(state['agent'].angle))
        direction_line.set_data([state['agent'].x, end_x], [state['agent'].y, end_y])
        
        # Get compass neuron activation
        activations = get_node_activations(network, genome, inputs, [best_neuron])
        activation = activations[best_neuron]
        
        # Update activation plot
        idx = int(state['agent'].angle) % 360
        current_point.set_data([idx], [activation])
        
        # Update compass visualization
        angle_rad = np.radians(state['agent'].angle)
        compass_marker.set_data([angle_rad], [0.8])
        compass_line.set_data([0, angle_rad], [0, 0.8])
        
        # Update text
        info_text.set_text(f'Angle: {state["agent"].angle:.1f}°\nActivation: {activation:.3f}')
        
        # Reset if completed or failed
        if done or state['agent'].returned_home:
            if state['agent'].returned_home:
                print(f"Successfully returned home after {state['agent'].steps_taken} steps!")
            state['agent'] = Agent((WIDTH/2, HEIGHT/2))
            state['reward_pos'] = reset_reward()
            reward.center = state['reward_pos']
            state['path'] = []
            path_line.set_data([], [])
        
        return agent_circle, direction_line, compass_marker, compass_line, current_point, info_text, path_line, reward
    
    # Create animation with explicit save_count
    anim = FuncAnimation(
        fig, update,
        init_func=init,
        interval=50,
        blit=True,
        save_count=1000  # Add this to prevent the warning
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    genome_path = "runs/run_20241229_004508/genomes/champion_genome_gen1600.pkl"
    visualize_agent_and_compass(genome_path)  # Call new visualization function