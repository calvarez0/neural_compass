# neat50/main.py

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


def randomize_reward_position(width, height, reward_size, min_distance=50, max_distance=150):
    """Generate reward position within valid distance range from center."""
    center_x, center_y = width // 2, height // 2

    while True:
        reward_x = random.randint(reward_size, width - reward_size)
        reward_y = random.randint(reward_size, height - reward_size)
        
        distance_from_center = math.sqrt((reward_x - center_x) ** 2 + (reward_y - center_y) ** 2)
        
        if min_distance <= distance_from_center <= max_distance:
            return (reward_x, reward_y)


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    num_trials = 3
    total_fitness = 0
    
    for _ in range(num_trials):
        beehive_pos = (WIDTH/2, HEIGHT/2)
        agent = Agent(beehive_pos)
        reward_pos = randomize_reward_position(WIDTH, HEIGHT, REWARD_SIZE, 75, 150)
        
        total_distance = 0
        last_pos = (agent.x, agent.y)
        movement_history = []
        completion_time = 0
        
        for step in range(1000):
            inputs = agent.get_sensor_data(reward_pos)
            outputs = net.activate(inputs)
            done = agent.update(outputs, reward_pos)
            
            current_pos = (agent.x, agent.y)
            dist_moved = distance(last_pos, current_pos)
            total_distance += dist_moved
            
            movement_history.append((agent.x, agent.y))
            if len(movement_history) > 20:
                movement_history.pop(0)
            
            last_pos = current_pos
            
            if done:
                completion_time = step
                break
            
            if agent.energy <= 0:
                break
        
        fitness = 0
        
        # Progressive distance-based reward
        if not agent.found_reward:
            dist_to_reward = distance((agent.x, agent.y), reward_pos)
            progress = 1.0 - (dist_to_reward / (math.sqrt(WIDTH**2 + HEIGHT**2)))
            fitness += 500 * progress ** 2  # Quadratic scaling for closer approaches
        
        # Achievement rewards with time bonus potential
        if agent.found_reward:
            fitness += 1000
            # Add time bonus for quick finding
            time_bonus = max(0, (500 - agent.steps_taken)) * 2
            fitness += time_bonus
            
        if agent.carrying_reward:
            fitness += 2000
            
        if agent.returned_home:
            fitness += 5000  # Increased base completion reward
            
            # Optimal path calculations with unlimited potential
            optimal_path = distance(beehive_pos, reward_pos) * 2
            path_ratio = total_distance / optimal_path
            
            # Exponential efficiency bonus that grows with better performance
            efficiency_score = math.exp(-2 * (path_ratio - 1.0))
            efficiency_bonus = 10000 * efficiency_score
            fitness += efficiency_bonus
            
            # Time completion bonus
            time_mult = max(1.0, (1000 - completion_time) / 100)
            fitness *= time_mult
            
            # Consecutive completion bonus (if you want to track this)
            # This would need to be stored between evaluations
            # fitness *= (1.0 + consecutive_completions * 0.1)
        
        # Energy bonus that scales with performance
        energy_ratio = agent.energy / agent.energy_default
        if energy_ratio > 0:
            # Energy efficiency multiplier that can grow unbounded
            energy_mult = 1.0 + (energy_ratio ** 2)  # Quadratic scaling
            fitness *= energy_mult
        
        # Step penalty that increases with time
        step_penalty = (agent.steps_taken / 10.0) ** 1.5
        fitness = max(0, fitness - step_penalty)
        
        total_fitness += fitness
    
    return total_fitness / num_trials


def run_neat_with_visualization(checkpoint_frequency=50, continue_frequency=500):
    """Run NEAT with periodic visualization and improved organization."""
    run_dir = create_run_directory()
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-neat")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    pop = neat.Population(config)
    
    # Setup statistics and reporting
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(False))
    
    best_genome = None
    best_fitness = -float('inf')
    generation = 0
    
    # Initialize statistics for generation 0
    genomes = list(pop.population.items())
    eval_genomes(genomes, config)
    stats.post_evaluate(config, pop, pop.species, pop.population)
    
    # Find best of initial population
    for _, g in genomes:
        if g.fitness > best_fitness:
            best_fitness = g.fitness
            best_genome = g
            save_genome(best_genome, config, generation, run_dir)
    
    while generation < 5000:
        for _ in range(checkpoint_frequency):
            generation += 1
            
            # Advance to next generation
            pop.population = pop.reproduction.reproduce(config, pop.species,
                                                     pop.config.pop_size,
                                                     generation)
            pop.species.speciate(config, pop.population, generation)
            
            # Evaluate all genomes
            genomes = list(pop.population.items())
            eval_genomes(genomes, config)
            
            # Update statistics
            stats.post_evaluate(config, pop, pop.species, pop.population)
            
            # Track best genome
            for _, g in genomes:
                if g.fitness > best_fitness:
                    best_fitness = g.fitness
                    best_genome = g
                    save_genome(best_genome, config, generation, run_dir)
        
        # Visualization at checkpoints
        if generation % checkpoint_frequency == 0:
            print(f"\nGeneration {generation}")
            print(f"Best Fitness: {best_fitness:.2f}")
            print(f"Number of species: {len(pop.species.species)}")
            print(f"Population size: {len(pop.population)}")
            
            # Find current generation's best genome
            current_best = None
            current_best_fitness = -float('inf')
            for _, g in genomes:
                if g.fitness > current_best_fitness:
                    current_best_fitness = g.fitness
                    current_best = g
            
            # Save the current generation's best genome
            save_genome(current_best, config, generation, run_dir, 
                    filename=f'champion_genome_gen{generation}.pkl')
            
            # Save the all-time best genome with a special name
            if current_best_fitness > best_fitness:
                save_genome(current_best, config, generation, run_dir,
                        filename='best_ever_genome.pkl')
            
            # Generate visualizations
            if stats.get_fitness_mean():
                plot_stats(stats, run_dir, filename=f'fitness_gen{generation}.svg')
                species_stats = stats.get_species_sizes()
                if species_stats:
                    plot_species(stats, run_dir, filename=f'speciation_gen{generation}.svg')
                    
            visualize_network_topology(current_best, config, generation, run_dir)
            save_agent_simulation(current_best, config, generation, run_dir)
            
            # Ask to continue at specified intervals
            if generation % continue_frequency == 0:
                response = input("\nContinue training? (y/n): ").lower().strip()
                if response != 'y':
                    break
    
    return run_dir

def main():
    run_neat_with_visualization(checkpoint_frequency=25, continue_frequency=500)

if __name__ == '__main__':
    main()