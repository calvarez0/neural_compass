import os
import neat
import math
import graphviz
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from utilities import *
from constants import *
from agent import *

def eval_genomes(genomes, config):
    """
    Evaluate a list of genomes.
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def randomize_reward_position(width, height, reward_size, min_distance_from_center=75):  # adjust distance from center here
    center_x, center_y = width // 2, height // 2

    while True:
        reward_x = random.randint(reward_size, width - reward_size)
        reward_y = random.randint(reward_size, height - reward_size)
        
        # Calculate the distance from the center
        distance_from_center = math.sqrt((reward_x - center_x) ** 2 + (reward_y - center_y) ** 2)
        
        # Check if it's outside the minimum distance
        if distance_from_center > min_distance_from_center:
            return (reward_x, reward_y)


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    num_trials = 3
    total_fitness = 0
    
    for _ in range(num_trials):
        # Random beehive position
        beehive_pos = (WIDTH/2, HEIGHT/2)
        
        agent = Agent(beehive_pos)
        
        # Random reward position with constraints
        while True:
            reward_pos = randomize_reward_position(WIDTH, HEIGHT, REWARD_SIZE)
            dist_to_home = distance(beehive_pos, reward_pos)
            if 50 <= dist_to_home <= 150:
                break
                        
        total_distance = 0
        last_pos = (agent.x, agent.y)
        movement_history = []
        last_angles = []
        
        while True:
            inputs = agent.get_sensor_data(reward_pos)
            outputs = net.activate(inputs)
            done = agent.update(outputs, reward_pos)
            
            # Calculate movement metrics
            current_pos = (agent.x, agent.y)
            dist_moved = distance(last_pos, current_pos)
            total_distance += dist_moved
            
            # Track rotation history
            last_angles.append(agent.angle)
            if len(last_angles) > 10:
                last_angles.pop(0)
                        
            # Track movement history
            movement_history.append((agent.x, agent.y))
            if len(movement_history) > 20:
                movement_history.pop(0)
                if len(movement_history) >= 3:
                    xs = [p[0] for p in movement_history]
                    ys = [p[1] for p in movement_history]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    if area < 2:
                        agent.energy -= 250  # Increased penalty 
            
            last_pos = current_pos
            
            if agent.energy <= 0 or done:
                break
        
        # Modified fitness calculation using agent's energy
        fitness = 0
        
        # Base rewards
        if agent.found_reward:
            fitness += 5000
        if agent.carrying_reward:
            fitness += 10000  # testing
        if agent.returned_home:
            fitness *= 5
            optimal_path_length = distance(agent.start_position, reward_pos) + distance(reward_pos, beehive_pos)
            if total_distance < optimal_path_length * 1.50:
                fitness *= 1000 # * (optimal_path_length / total_distance) \
            elif total_distance < optimal_path_length * 1.75:
                fitness *= 650
            elif total_distance < optimal_path_length * 2:
                fitness *= 250
        
        # Energy efficiency bonus
        fitness += agent.energy  # Use agent's energy instead of separate energy variable
        
        # Distance penalties
        if not agent.found_reward:
            dist_to_reward = distance((agent.x, agent.y), reward_pos)
            fitness -= dist_to_reward * 3
        if agent.carrying_reward and not agent.returned_home:
            dist_to_home = distance((agent.x, agent.y), beehive_pos)
            fitness -= dist_to_home * 4  # changed from 4 to 40
            
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
    
    while generation < 5000:  # Total generations
        # Run evolution for checkpoint_frequency generations
        for _ in range(checkpoint_frequency):
            generation += 1
            
            # Evaluate all genomes
            genomes = list(pop.population.items())
            for _, g in genomes:
                g.fitness = eval_genome(g, config)
                if g.fitness > best_fitness:
                    best_fitness = g.fitness
                    best_genome = g
            
            # Advance to the next generation
            if generation < 5000:  # Don't advance on final generation
                pop.population = pop.reproduction.reproduce(config, pop.species,
                                                         pop.config.pop_size,
                                                         generation)
                pop.species.speciate(config, pop.population, generation)
        
        # Plot the topology of the champion every 100 generations
        if generation % 50 == 0:
            print(f"Plotting champion topology at generation {generation}")
            visualize_network_topology(best_genome, config, generation)
            save_agent_simulation(neat.nn.FeedForwardNetwork.create(best_genome, config), config, generation)

        # Visualize current best
        print(f"\nGeneration {generation}")
        print(f"Best Fitness: {best_fitness:.2f}")
        
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        visualize_agent(net, config, max_steps=1000)
        
        response = input("\nContinue training? (y/n): ").lower().strip()
        if response != 'y':
            break
    
    return best_genome, config

def main():
    # Run NEAT with periodic visualization
    winner, config = run_neat_with_visualization(checkpoint_frequency=50)
    
    # Final visualization of the best agent
    print("\nFinal visualization of best agent:")
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    visualize_agent(winner_net, config)

if __name__ == '__main__':
    main()