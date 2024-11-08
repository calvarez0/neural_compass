import os
import neat
import math
import graphviz
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from constants import *
from agent import *
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import colorsys

def visualize_network_topology(genome, config, generation, figsize=(12, 8)):
    """
    Creates an enhanced visualization of a NEAT neural network topology.

    Args:
        genome: NEAT genome to visualize
        config: NEAT configuration object
        generation: Current generation number
        figsize: Tuple of figure dimensions (width, height)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    # Get network dimensions
    num_inputs = config.genome_config.num_inputs
    num_outputs = config.genome_config.num_outputs
    num_hidden = len([n for n in genome.nodes.keys() 
                     if n >= num_inputs + num_outputs])

    # Calculate node positions
    positions = calculate_node_positions(genome, num_inputs, num_outputs, num_hidden)

    # Create custom colormap for connections
    colors = [(0.7, 0, 0), (0.9, 0.9, 0.9), (0, 0.7, 0)]  # red -> light gray -> green
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    # Draw connections first (so they're behind nodes)
    max_weight = max(abs(c.weight) for c in genome.connections.values() if c.enabled)
    draw_connections(ax, genome, positions, max_weight, custom_cmap)

    # Draw nodes
    node_labels = draw_nodes(ax, genome, positions, num_inputs, num_outputs)

    # Add annotations and styling
    add_annotations(ax, generation, node_labels)

    # Save figure
    filename = f'champion_topology_gen{generation}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network topology saved as '{filename}'")

def calculate_node_positions(genome, num_inputs, num_outputs, num_hidden):
    """Calculate positions for all nodes using a layered layout algorithm."""
    positions = {}
    
    # Get all node IDs from both genome nodes and connections
    all_nodes = set(genome.nodes.keys())
    for conn in genome.connections.values():
        all_nodes.add(conn.key[0])
        all_nodes.add(conn.key[1])
    
    # Handle bias nodes (any negative indices)
    bias_nodes = sorted([n for n in all_nodes if n < 0])
    if bias_nodes:
        bias_spacing = 1.0 / (len(bias_nodes) + 1)
        for i, node_id in enumerate(bias_nodes):
            positions[node_id] = np.array([0.1, (i + 1) * bias_spacing * 0.3])
    
    # Position input nodes above bias nodes
    input_spacing = 1.0 / (num_inputs + 1)
    for i in range(num_inputs):
        positions[i] = np.array([0.1, 0.3 + (i + 1) * input_spacing * 0.7])
    
    # Position output nodes
    output_spacing = 1.0 / (num_outputs + 1)
    for i in range(num_outputs):
        node_id = num_inputs + i
        positions[node_id] = np.array([0.9, (i + 1) * output_spacing])
    
    # Sort and position hidden nodes
    hidden_nodes = sorted([n for n in all_nodes 
                         if n >= num_inputs + num_outputs])
    
    hidden_layers = sort_hidden_layers(genome, hidden_nodes, num_inputs, positions)
    
    if hidden_layers:
        layer_spacing = 0.8 / (len(hidden_layers) + 1)
        for layer_idx, layer in enumerate(hidden_layers):
            node_spacing = 1.0 / (len(layer) + 1)
            x_pos = 0.1 + (layer_idx + 1) * layer_spacing
            for node_idx, node_id in enumerate(layer):
                y_pos = (node_idx + 1) * node_spacing
                positions[node_id] = np.array([x_pos, y_pos])
    
    return positions

def sort_hidden_layers(genome, hidden_nodes, num_inputs, positions):
    """Sort hidden nodes into layers based on their connectivity."""
    layers = []
    remaining_nodes = set(hidden_nodes)
    
    # Include bias nodes and input nodes in the input set
    input_nodes = set(k for k in positions.keys() if k < num_inputs)
    
    while remaining_nodes:
        current_layer = set()
        for node in remaining_nodes:
            input_connections = set()
            for conn in genome.connections.values():
                if conn.enabled and conn.key[1] == node:
                    input_connections.add(conn.key[0])
            
            # Check if all inputs to this node are from previous layers or inputs
            if input_connections and input_connections.issubset(input_nodes):
                current_layer.add(node)
        
        if not current_layer:
            # If no nodes found, put remaining nodes in one layer
            layers.append(sorted(list(remaining_nodes)))
            break
            
        layers.append(sorted(list(current_layer)))
        remaining_nodes -= current_layer
        input_nodes.update(current_layer)
    
    return layers

def draw_connections(ax, genome, positions, max_weight, cmap):
    """Draw neural network connections with weight-based styling."""
    # Sort connections by weight for proper layering
    connections = sorted(
        genome.connections.values(),
        key=lambda x: abs(x.weight)
    )
    
    for conn_gene in connections:
        if conn_gene.enabled:
            if conn_gene.key[0] not in positions or conn_gene.key[1] not in positions:
                continue
                
            start = positions[conn_gene.key[0]]
            end = positions[conn_gene.key[1]]
            
            # Normalize weight for color mapping
            weight_normalized = (conn_gene.weight + max_weight) / (2 * max_weight)
            color = cmap(weight_normalized)
            
            # Calculate arrow properties based on weight
            width = 1 + 2 * abs(conn_gene.weight) / max_weight
            
            # Adjust curvature based on node positions
            rad = 0.2
            if abs(start[0] - end[0]) < 0.1:  # Nearly vertical connections
                rad = 0.4
            
            # Create curved arrow
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='-|>',
                connectionstyle=f'arc3,rad={rad}',
                mutation_scale=10,
                linewidth=width,
                color=color,
                alpha=0.6,
                zorder=1
            )
            ax.add_patch(arrow)
            
            # Add weight label with improved positioning
            mid_point = (start + end) / 2
            offset = np.array([0.02, 0.02])
            plt.annotate(
                f'{conn_gene.weight:.2f}',
                xy=mid_point,
                xytext=mid_point + offset,
                fontsize=6,
                alpha=0.7,
                zorder=3
            )

def draw_nodes(ax, genome, positions, num_inputs, num_outputs):
    """Draw neural network nodes with labels."""
    node_labels = {}

    for node_id, pos in positions.items():
        if node_id < 0:
            color = '#FFB6C1'
            size = 0.03
            label = f'Bias {node_id}'
        elif node_id < num_inputs:
            color = '#AED6F1'
            size = 0.03
            label = f'In {node_id}'
        elif node_id < num_inputs + num_outputs:
            color = '#A2D9A2'
            size = 0.03
            label = f'Out {node_id}'
        else:
            color = '#F5CBA7'
            size = 0.025
            label = f'H {node_id}'

        circle = Circle(
            pos,
            size,
            facecolor=color,
            edgecolor='gray',
            linewidth=1,
            alpha=0.8,
            zorder=2
        )
        ax.add_patch(circle)
        plt.annotate(
            label,
            xy=pos,
            xytext=(pos[0], pos[1] - 0.02),
            ha='center',
            va='top',
            fontsize=8
        )
        node_labels[node_id] = label

    return node_labels

def add_annotations(ax, generation, node_labels):
    """Add title and other annotations to the visualization."""
    plt.title(f'Champion Network Topology - Generation {generation}', 
              pad=20, size=14, weight='bold')

    legend_elements = [
        Circle((0, 0), 0.02, facecolor='#FFB6C1', label='Bias'),
        Circle((0, 0), 0.02, facecolor='#AED6F1', label='Input'),
        Circle((0, 0), 0.02, facecolor='#A2D9A2', label='Output'),
        Circle((0, 0), 0.02, facecolor='#F5CBA7', label='Hidden')
    ]
    ax.legend(handles=legend_elements, 
             loc='center left', 
             bbox_to_anchor=(1, 0.5))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')


def visualize_agent(net, config, max_steps=None):
    class AnimationState:
        def __init__(self):
            self.running = True
            self.steps = 0
            self.beehive_pos = (WIDTH/2, HEIGHT/2)
            self.agent = Agent(self.beehive_pos)
            
            while True:
                self.reward_pos = (
                    random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE),
                    random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE)
                )
                dist_to_home = distance(self.beehive_pos, self.reward_pos)
                if 50 <= dist_to_home <= 150:
                    break

    state = AnimationState()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def on_close(event):
        state.running = False
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    def update():
        if not state.running or (max_steps and state.steps >= max_steps):
            return False
        
        state.steps += 1
        ax.clear()
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        
        # Get neural network output and update agent
        inputs = state.agent.get_sensor_data(state.reward_pos)
        outputs = net.activate(inputs)
        done = state.agent.update(outputs, state.reward_pos)
        
        # Draw path
        if len(state.agent.path_positions) > 1:
            path_x = [p[0] for p in state.agent.path_positions]
            path_y = [p[1] for p in state.agent.path_positions]
            ax.plot(path_x, path_y, 'b-', alpha=0.5)
        
        # Draw vision cone
        if state.agent.vision_active:
            end_x = state.agent.x + VISION_RANGE * math.cos(math.radians(state.agent.angle))
            end_y = state.agent.y + VISION_RANGE * math.sin(math.radians(state.agent.angle))
            ax.plot([state.agent.x, end_x], [state.agent.y, end_y], 'g-', alpha=0.3)
        
        # Draw agent
        agent_circle = patches.Circle(
            (state.agent.x, state.agent.y),
            AGENT_SIZE,
            color='red' if state.agent.carrying_reward else 'green'
        )
        ax.add_patch(agent_circle)
        
        # Draw direction indicator
        end_x = state.agent.x + AGENT_SIZE * 2 * math.cos(math.radians(state.agent.angle))
        end_y = state.agent.y + AGENT_SIZE * 2 * math.sin(math.radians(state.agent.angle))
        ax.plot([state.agent.x, end_x], [state.agent.y, end_y], 'k-', linewidth=2)
        
        # Draw beehive and reward
        beehive_rect = patches.Rectangle(
            (state.beehive_pos[0] - BEEHIVE_SIZE//2, state.beehive_pos[1] - BEEHIVE_SIZE//2),
            BEEHIVE_SIZE, BEEHIVE_SIZE, color='brown'
        )
        ax.add_patch(beehive_rect)
        
        reward_circle = patches.Circle(
            (state.reward_pos[0], state.reward_pos[1]),
            REWARD_SIZE,
            color='red'
        )
        ax.add_patch(reward_circle)
        
        # Add status text
        status_text = [
            f"Steps: {state.agent.steps_taken}",
            f"Vision: {'ON' if state.agent.vision_active else 'OFF'}",
            f"Carrying Reward: {'YES' if state.agent.carrying_reward else 'NO'}",
            f"Energy: {state.agent.energy}",  # Display energy level
        ]
        
        current_target = state.beehive_pos if state.agent.carrying_reward else state.reward_pos
        target_dist = distance((state.agent.x, state.agent.y), current_target)
        status_text.append(f"Distance to {'Home' if state.agent.carrying_reward else 'Reward'}: {target_dist:.1f}")
        
        for i, text in enumerate(status_text):
            ax.text(5, HEIGHT - 5 - (i * 15), text,
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.7))
        
        if done:
            return False
        
        ax.set_aspect('equal')
        plt.draw()
        return True
    
    while plt.fignum_exists(fig.number):
        if not update():
            break
        plt.pause(0.05)
    
    plt.close(fig)

def save_agent_simulation(net, config, generation, max_steps=500, filename=None):
    """Save agent simulation as MP4 video."""
    if filename is None:
        filename = f'agent_simulation_gen{generation}.mp4'
    
    class SimState:
        def __init__(self):
            self.steps = 0
            self.beehive_pos = (WIDTH/2, HEIGHT/2)
            self.agent = Agent(self.beehive_pos)
            
            while True:
                self.reward_pos = (
                    random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE),
                    random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE)
                )
                dist_to_home = distance(self.beehive_pos, self.reward_pos)
                if 50 <= dist_to_home <= 150:
                    break

    state = SimState()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def animation_frame(frame):
        if state.steps >= max_steps:
            return []
        
        state.steps += 1
        ax.clear()
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        
        # Get neural network output and update agent
        inputs = state.agent.get_sensor_data(state.reward_pos)
        outputs = net.activate(inputs)
        done = state.agent.update(outputs, state.reward_pos)
        
        artists = []
        
        # Draw path
        if len(state.agent.path_positions) > 1:
            path_x = [p[0] for p in state.agent.path_positions]
            path_y = [p[1] for p in state.agent.path_positions]
            line, = ax.plot(path_x, path_y, 'b-', alpha=0.5)
            artists.append(line)
        
        # Draw vision cone
        if state.agent.vision_active:
            end_x = state.agent.x + VISION_RANGE * math.cos(math.radians(state.agent.angle))
            end_y = state.agent.y + VISION_RANGE * math.sin(math.radians(state.agent.angle))
            line, = ax.plot([state.agent.x, end_x], [state.agent.y, end_y], 'g-', alpha=0.3)
            artists.append(line)
        
        # Draw agent
        agent_circle = patches.Circle(
            (state.agent.x, state.agent.y),
            AGENT_SIZE,
            color='red' if state.agent.carrying_reward else 'green'
        )
        ax.add_patch(agent_circle)
        artists.append(agent_circle)
        
        # Draw direction indicator
        end_x = state.agent.x + AGENT_SIZE * 2 * math.cos(math.radians(state.agent.angle))
        end_y = state.agent.y + AGENT_SIZE * 2 * math.sin(math.radians(state.agent.angle))
        line, = ax.plot([state.agent.x, end_x], [state.agent.y, end_y], 'k-', linewidth=2)
        artists.append(line)
        
        # Draw beehive and reward
        beehive_rect = patches.Rectangle(
            (state.beehive_pos[0] - BEEHIVE_SIZE//2, state.beehive_pos[1] - BEEHIVE_SIZE//2),
            BEEHIVE_SIZE, BEEHIVE_SIZE, color='brown'
        )
        ax.add_patch(beehive_rect)
        artists.append(beehive_rect)
        
        reward_circle = patches.Circle(
            (state.reward_pos[0], state.reward_pos[1]),
            REWARD_SIZE,
            color='red'
        )
        ax.add_patch(reward_circle)
        artists.append(reward_circle)
        
        # Add status text
        status_text = [
            f"Steps: {state.agent.steps_taken}",
            f"Vision: {'ON' if state.agent.vision_active else 'OFF'}",
            f"Carrying Reward: {'YES' if state.agent.carrying_reward else 'NO'}",
            f"Energy: {state.agent.energy}",
        ]
        
        current_target = state.beehive_pos if state.agent.carrying_reward else state.reward_pos
        target_dist = distance((state.agent.x, state.agent.y), current_target)
        status_text.append(f"Distance to {'Home' if state.agent.carrying_reward else 'Reward'}: {target_dist:.1f}")
        
        for i, text in enumerate(status_text):
            text_artist = ax.text(5, HEIGHT - 5 - (i * 15), text,
                                verticalalignment='top',
                                bbox=dict(facecolor='white', alpha=0.7))
            artists.append(text_artist)
        
        ax.set_aspect('equal')
        
        if done:
            state.steps = max_steps  # End the animation
            
        return artists
    
    # Create animation
    anim = FuncAnimation(fig, animation_frame, frames=max_steps,
                        interval=50, blit=True)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=20, bitrate=2000)
    anim.save(filename, writer=writer)
    plt.close()
    print(f"Simulation saved as '{filename}'")