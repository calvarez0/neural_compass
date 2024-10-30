import os
import neat
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Constants
WIDTH, HEIGHT = 200, 200
AGENT_SIZE = 10
REWARD_SIZE = 10
BEEHIVE_SIZE = 20
VISION_RANGE = 200
ROTATION_SPEED = 20  # Increased from 10
MOVE_SPEED = 10     # Increased from 5
MAX_STEPS = 500     # Reduced from 1000 to make episodes shorter


def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class Agent:
    def __init__(self, beehive_pos, start_pos=None):
        self.beehive_position = beehive_pos
        self.start_position = start_pos if start_pos else beehive_pos
        self.reset()
        
    def reset(self):
        # Start from start position instead of beehive
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.angle = random.randint(0, 360)
        self.vision_active = True
        self.reward_position = None
        self.carrying_reward = False
        self.path_positions = [(self.x, self.y)]
        self.steps_taken = 0
        self.found_reward = False
        self.returned_home = False
        
    def get_sensor_data(self, reward_pos):
        # Simplified and more direct sensor data
        dx_reward = (reward_pos[0] - self.x) / WIDTH if self.vision_active else 0
        dy_reward = (reward_pos[1] - self.y) / HEIGHT if self.vision_active else 0
        
        dx_home = (self.beehive_position[0] - self.x) / WIDTH
        dy_home = (self.beehive_position[1] - self.y) / HEIGHT
        
        # Simplified angle calculations
        reward_angle = math.degrees(math.atan2(dy_reward, dx_reward)) % 360
        home_angle = math.degrees(math.atan2(dy_home, dx_home)) % 360
        
        relative_reward_angle = ((reward_angle - self.angle) % 360) / 360
        relative_home_angle = ((home_angle - self.angle) % 360) / 360
        
        return [
            dx_reward, dy_reward,    # Vector to reward
            dx_home, dy_home,        # Vector to home
            relative_reward_angle,    # Angle to reward
            relative_home_angle,      # Angle to home
            1.0 if self.carrying_reward else 0.0,  # Carrying state
            1.0 if self.vision_active else 0.0     # Vision state
        ]
    
    def update(self, action, reward_pos):
        # Interpret neural network outputs
        rotate = action[0]  # -1 to 1
        move_forward = action[1] > 0  # Binary decision
        
        # Apply rotation
        self.angle = (self.angle + rotate * ROTATION_SPEED) % 360
        
        # Move forward if indicated
        if move_forward:
            dx = math.cos(math.radians(self.angle)) * MOVE_SPEED
            dy = math.sin(math.radians(self.angle)) * MOVE_SPEED
            new_x = self.x + dx
            new_y = self.y + dy
            
            # Boundary checking
            if 0 <= new_x <= WIDTH:
                self.x = new_x
            if 0 <= new_y <= HEIGHT:
                self.y = new_y
                
        self.path_positions.append((self.x, self.y))
        self.steps_taken += 1
        
        # Check if reward is found
        if self.vision_active and distance((self.x, self.y), reward_pos) <= VISION_RANGE:
            self.reward_position = reward_pos
            if not self.found_reward:
                self.found_reward = True
        
        # Check if reward is picked up
        if self.reward_position and not self.carrying_reward:
            if distance((self.x, self.y), reward_pos) <= AGENT_SIZE + REWARD_SIZE:
                self.carrying_reward = True
                self.vision_active = False  # Turn off vision after picking up reward
        
        # Check if returned home with reward
        if self.carrying_reward and distance((self.x, self.y), self.beehive_position) <= AGENT_SIZE + BEEHIVE_SIZE:
            self.returned_home = True
            return True
            
        # Check if maximum steps exceeded
        if self.steps_taken >= MAX_STEPS:
            return True
            
        return False

def eval_genomes(genomes, config):
    """
    Evaluate a list of genomes.
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    num_trials = 3
    total_fitness = 0
    
    for _ in range(num_trials):
        # Random beehive position
        beehive_pos = (
            random.randint(BEEHIVE_SIZE, WIDTH - BEEHIVE_SIZE),
            random.randint(BEEHIVE_SIZE, HEIGHT - BEEHIVE_SIZE)
        )
        
        # Random start position (different from beehive)
        while True:
            start_pos = (
                random.randint(AGENT_SIZE, WIDTH - AGENT_SIZE),
                random.randint(AGENT_SIZE, HEIGHT - AGENT_SIZE)
            )
            if distance(start_pos, beehive_pos) >= 50:  # Minimum distance from beehive
                break
                
        agent = Agent(beehive_pos, start_pos)
        
        # Random reward position with constraints
        while True:
            reward_pos = (
                random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE),
                random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE)
            )
            dist_to_home = distance(beehive_pos, reward_pos)
            dist_to_start = distance(start_pos, reward_pos)
            if 50 <= dist_to_home <= 150 and 50 <= dist_to_start <= 150:
                break
        
        # Increase circular motion penalty
        energy = 1000
        total_distance = 0
        last_pos = (agent.x, agent.y)
        movement_history = []
        last_angles = []  # Track recent rotation angles
        
        while True:
            inputs = agent.get_sensor_data(reward_pos)
            outputs = net.activate(inputs)
            done = agent.update(outputs, reward_pos)
            
            # Calculate movement costs
            current_pos = (agent.x, agent.y)
            dist_moved = distance(last_pos, current_pos)
            total_distance += dist_moved
            
            # Track rotation history
            last_angles.append(agent.angle)
            if len(last_angles) > 10:
                last_angles.pop(0)
            
            # Stronger penalties for circular motion
            if len(last_angles) >= 10:
                angle_diff = abs(last_angles[-1] - last_angles[0])
                if angle_diff > 330 or angle_diff < 30:  # Almost complete circle
                    energy -= 50  # Much stronger penalty
            
            # Basic movement energy costs
            energy -= dist_moved
            if agent.carrying_reward:
                energy -= dist_moved * 2
            
            movement_history.append((agent.x, agent.y))
            if len(movement_history) > 20:
                movement_history.pop(0)
                if len(movement_history) >= 3:
                    xs = [p[0] for p in movement_history]
                    ys = [p[1] for p in movement_history]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    if area < 100:
                        energy -= 20  # Increased penalty
            
            last_pos = current_pos
            
            if energy <= 0 or done:
                break
        
        # Modified fitness calculation
        fitness = 0
        
        # Base rewards
        if agent.found_reward:
            fitness += 500
        if agent.carrying_reward:
            fitness += 1000
        if agent.returned_home:
            fitness += 2000
            optimal_path_length = distance(agent.start_position, reward_pos) + distance(reward_pos, beehive_pos)
            if total_distance < optimal_path_length * 1.5:
                fitness += 1000 * (optimal_path_length / total_distance)
        
        # Energy efficiency and movement penalties
        fitness += energy
        
        # Distance penalties
        if not agent.found_reward:
            dist_to_reward = distance((agent.x, agent.y), reward_pos)
            fitness -= dist_to_reward * 3
        if agent.carrying_reward and not agent.returned_home:
            dist_to_home = distance((agent.x, agent.y), beehive_pos)
            fitness -= dist_to_home * 4
            
        total_fitness += fitness
    
    return total_fitness / num_trials

def run_neat_with_visualization(checkpoint_frequency=50):
    """Run NEAT with periodic visualization of the best agent"""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-neat")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(False))  # Set to False to reduce output
    
    best_genome = None
    best_fitness = -float('inf')
    generation = 0
    
    while generation < 500:  # Total generations
        # Run evolution for checkpoint_frequency generations
        for _ in range(checkpoint_frequency):
            generation += 1
            
            # Evaluate all genomes
            genomes = list(pop.population.items())  # Changed from iteritems to items()
            for _, g in genomes:
                g.fitness = eval_genome(g, config)
                if g.fitness > best_fitness:
                    best_fitness = g.fitness
                    best_genome = g
            
            # Advance to the next generation
            if generation < 500:  # Don't advance on final generation
                pop.population = pop.reproduction.reproduce(config, pop.species,
                                                         pop.config.pop_size,
                                                         generation)
                pop.species.speciate(config, pop.population, generation)
        
        # Visualize current best
        print(f"\nGeneration {generation}")
        print(f"Best Fitness: {best_fitness:.2f}")
        
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        visualize_agent(net, config, max_steps=200)
        
        response = input("\nContinue training? (y/n): ").lower().strip()
        if response != 'y':
            break
    
    return best_genome, config

def visualize_agent(net, config, max_steps=None):
    class AnimationState:
        def __init__(self):
            self.running = True
            self.steps = 0
            
            # Random beehive position
            self.beehive_pos = (
                random.randint(BEEHIVE_SIZE, WIDTH - BEEHIVE_SIZE),
                random.randint(BEEHIVE_SIZE, HEIGHT - BEEHIVE_SIZE)
            )
            
            # Random start position
            while True:
                start_pos = (
                    random.randint(AGENT_SIZE, WIDTH - AGENT_SIZE),
                    random.randint(AGENT_SIZE, HEIGHT - AGENT_SIZE)
                )
                if distance(start_pos, self.beehive_pos) >= 50:
                    break
                    
            self.agent = Agent(self.beehive_pos, start_pos)
            
            # Random reward position
            while True:
                self.reward_pos = (
                    random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE),
                    random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE)
                )
                dist_to_home = distance(self.beehive_pos, self.reward_pos)
                dist_to_start = distance(start_pos, self.reward_pos)
                if 50 <= dist_to_home <= 150 and 50 <= dist_to_start <= 150:
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
        status_text = []
        status_text.append(f"Steps: {state.agent.steps_taken}")
        status_text.append("Vision: ON" if state.agent.vision_active else "Vision: OFF")
        status_text.append("Carrying Reward: YES" if state.agent.carrying_reward else "Carrying Reward: NO")
        
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
    
    # Main animation loop
    while plt.fignum_exists(fig.number):
        if not update():
            break
        plt.pause(0.05)  # 50ms delay, similar to previous interval
    
    plt.close(fig)

def main():
    # Run NEAT with periodic visualization
    winner, config = run_neat_with_visualization(checkpoint_frequency=50)
    
    # Final visualization of the best agent
    print("\nFinal visualization of best agent:")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    visualize_agent(winner_net, config)

if __name__ == '__main__':
    main()