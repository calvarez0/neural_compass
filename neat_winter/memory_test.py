import os
import neat
import math
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib import colors
import pickle
from agent import Agent, distance
from constants import *
import random

def load_genome(filepath):
    """Load a genome from a pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['genome'], data['config'], data['generation']

def run_simulation(genome_path):
    """Run continuous simulation with memory grid visualization."""
    # Load genome
    genome, config, generation = load_genome(genome_path)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121)  # Simulation
    ax2 = fig.add_subplot(122)  # Memory grid visualization
    
    # Initialize simulation state
    state = {
        'agent': Agent((WIDTH/2, HEIGHT/2)),
        'reward_pos': None,
        'frame': 0,
        'path_line': None
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
    
    # Create static patches for simulation view
    agent_circle = patches.Circle((state['agent'].x, state['agent'].y), AGENT_SIZE, color='green')
    ax1.add_patch(agent_circle)
    
    direction_line, = ax1.plot([0, 0], [0, 0], 'k-', linewidth=2)
    
    beehive_rect = patches.Rectangle(
        (WIDTH/2 - BEEHIVE_SIZE/2, HEIGHT/2 - BEEHIVE_SIZE/2),
        BEEHIVE_SIZE, BEEHIVE_SIZE, color='brown'
    )
    ax1.add_patch(beehive_rect)
    
    reward_circle = patches.Circle(state['reward_pos'], REWARD_SIZE, color='red')
    ax1.add_patch(reward_circle)
    
    # Set up status text
    status_texts = []
    for i in range(3):
        text = ax1.text(5, HEIGHT - 5 - (i * 15), "", verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.7))
        status_texts.append(text)
    
    # Set up axis limits for simulation view
    ax1.set_xlim(0, WIDTH)
    ax1.set_ylim(0, HEIGHT)
    
    # Initialize memory grid visualization
    memory_grid_img = ax2.imshow(np.zeros((3, 3)), 
                                cmap='YlOrRd', 
                                interpolation='nearest',
                                vmin=0, vmax=1)
    ax2.set_title("Local Memory Grid")
    
    # Add colorbar
    plt.colorbar(memory_grid_img, ax=ax2, label='Memory Strength')
    
    # Call tight_layout once before animation starts
    plt.tight_layout()
    
    def update(frame):
        artists = []
        
        # Run multiple simulation steps per frame
        for _ in range(3):
            state['frame'] = frame
            
            # Get neural network input and update agent
            inputs = state['agent'].get_sensor_data(state['reward_pos'])
            outputs = net.activate(inputs)
            done = state['agent'].update(outputs, state['reward_pos'])
            
            # Reset if task completed or failed
            if done or state['agent'].returned_home:
                if state['agent'].returned_home:
                    print(f"Successfully returned home after {state['agent'].steps_taken} steps!")
                state['agent'] = Agent((WIDTH/2, HEIGHT/2))
                state['reward_pos'] = reset_reward()
                reward_circle.center = state['reward_pos']
                artists.append(reward_circle)
        
        # Update simulation visualization elements
        agent_circle.center = (state['agent'].x, state['agent'].y)
        agent_circle.set_facecolor('red' if state['agent'].carrying_reward else 'green')
        artists.append(agent_circle)
        
        reward_circle.center = state['reward_pos']
        artists.append(reward_circle)
        
        # Update direction indicator
        end_x = state['agent'].x + AGENT_SIZE * 2 * math.cos(math.radians(state['agent'].angle))
        end_y = state['agent'].y + AGENT_SIZE * 2 * math.sin(math.radians(state['agent'].angle))
        direction_line.set_data([state['agent'].x, end_x], [state['agent'].y, end_y])
        artists.append(direction_line)
        
        # Update path
        if len(state['agent'].path_positions) > 1:
            path_x = [p[0] for p in state['agent'].path_positions[-150:]]
            path_y = [p[1] for p in state['agent'].path_positions[-150:]]
            if state['path_line'] is None:
                state['path_line'], = ax1.plot(path_x, path_y, 'b-', alpha=0.5)
            else:
                state['path_line'].set_data(path_x, path_y)
            artists.append(state['path_line'])
        
        # Update status text
        status_text = [
            f"Steps: {state['agent'].steps_taken}",
            f"Energy: {state['agent'].energy:.0f}",
            f"Carrying: {'YES' if state['agent'].carrying_reward else 'NO'}"
        ]
        for text, txt in zip(status_texts, status_text):
            text.set_text(txt)
            artists.append(text)
        
        # Update memory grid visualization
        memory_values = state['agent'].get_local_memory()
        memory_grid = np.array(memory_values).reshape(3, 3)
        # Flip the grid vertically to match the simulation's coordinate system
        memory_grid = np.flipud(memory_grid)
        memory_grid_img.set_array(memory_grid)
        artists.append(memory_grid_img)
        
        return artists
    
    # Create animation with proper blitting
    anim = FuncAnimation(fig, update, interval=150, blit=True)
    plt.show()

if __name__ == "__main__":
    # Specify the path to your genome file
    genome_path = "runs/run_20241227_094534/genomes/champion_genome_gen3450.pkl"  # Update this path
    run_simulation(genome_path)