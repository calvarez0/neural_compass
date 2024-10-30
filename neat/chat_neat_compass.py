import neat
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Constants and agent parameters
WIDTH, HEIGHT = 200, 200
AGENT_SIZE = 10
REWARD_SIZE = 10
VISION_RANGE = 200
ROTATION_SPEED = 10
MOVE_SPEED = 5

class NEATAgent:
    def __init__(self, genome, config, beehive_position):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.beehive_position = beehive_position
        self.reset()

    def reset(self):
        # Initialize agent's position, angle, and vision
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.angle = random.uniform(0, 360)
        self.has_reward = False
        self.reward_position = None
        self.steps_taken = 0

    def step(self):
        inputs = self.get_inputs()
        output = self.net.activate(inputs)
        action = self.interpret_output(output)
        self.perform_action(action)
        self.steps_taken += 1

    def get_inputs(self):
        # Inputs: x, y position, angle, beehive vector (simplified for demo)
        dx = self.beehive_position[0] - self.x
        dy = self.beehive_position[1] - self.y
        return [self.x / WIDTH, self.y / HEIGHT, self.angle / 360, dx / WIDTH, dy / HEIGHT]

    def interpret_output(self, output):
        # Interpret NEAT output (e.g., 3 actions: move forward, rotate left, rotate right)
        return np.argmax(output)  # Assume output is [move_forward_prob, rotate_left_prob, rotate_right_prob]

    def perform_action(self, action):
        if action == 0:  # Move forward
            dx = MOVE_SPEED * math.cos(math.radians(self.angle))
            dy = MOVE_SPEED * math.sin(math.radians(self.angle))
            self.x = max(0, min(WIDTH, self.x + dx))
            self.y = max(0, min(HEIGHT, self.y + dy))
        elif action == 1:  # Rotate left
            self.angle = (self.angle - ROTATION_SPEED) % 360
        elif action == 2:  # Rotate right
            self.angle = (self.angle + ROTATION_SPEED) % 360

    def get_distance_to_beehive(self):
        return math.sqrt((self.x - self.beehive_position[0]) ** 2 + (self.y - self.beehive_position[1]) ** 2)

def eval_genome(genome, config):
    beehive_position = (WIDTH // 2, HEIGHT // 2)
    agent = NEATAgent(genome, config, beehive_position)
    fitness = 0

    for _ in range(1000):  # Max steps
        agent.step()
        # Fitness: Reward for moving towards goal, penalty for unnecessary movement
        fitness += 1 / (1 + agent.get_distance_to_beehive())
        
        if agent.get_distance_to_beehive() < AGENT_SIZE:
            fitness += 100  # Bonus for reaching beehive
            break

    genome.fitness = fitness

def run_neat():
    config_path = "./neat-config"  # Path to your NEAT config file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genome, 50)
    return winner, config

if __name__ == "__main__":
    winner, config = run_neat()
    print("Best genome:", winner)
