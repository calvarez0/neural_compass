import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Constants
WIDTH, HEIGHT = 200,200
AGENT_SIZE = 10
REWARD_SIZE = 10
BEEHIVE_SIZE = 20
VISION_RANGE = 200
ROTATION_SPEED = 10
MOVE_SPEED = 5

# Training parameters
TRAINING_EPISODES = 100000  # Increased for better learning
MAX_STEPS_PER_EPISODE = 1000  # Shorter episodes
GRID_SIZE = 4 # Reduced for better state generalization

# RL parameters
LEARNING_RATE = 0.005 # Reduced to prevent overshooting
DISCOUNT_FACTOR = 0.999
  # Increased for long-term planning
EPSILON_START = 1.0
EPSILON_END = 0.5  # Higher minimum exploration
EPSILON_DECAY = 0.99  # Slower decay

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

class TrainingAgent:
    def __init__(self, beehive_pos):  # Changed from (x, y)
        self.beehive_position = beehive_pos
        self.grid_size = GRID_SIZE
        self.q_table = {}
        self.actions = ['forward', 'rotate_left', 'rotate_right']
        self.epsilon = EPSILON_START
        self.x = beehive_pos[0]
        self.y = beehive_pos[1]
        self.angle = random.randint(0, 360)
        self.reset()


    def reset(self):
        # Start from random position
        angle_to_beehive = random.uniform(0, 2 * math.pi)
        distance_to_beehive = random.uniform(100, 400)
        
        self.x = self.beehive_position[0] + distance_to_beehive * math.cos(angle_to_beehive)
        self.y = self.beehive_position[1] + distance_to_beehive * math.sin(angle_to_beehive)
        
        # Constrain to boundaries
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))
        
        self.angle = random.randint(0, 360)
        self.prev_distance = self.get_distance_to_beehive()
        return self.get_state()

    def get_distance_to_beehive(self):
        return distance((self.x, self.y), self.beehive_position)

    def get_state(self):
        grid_x = int(self.x / (WIDTH / self.grid_size))
        grid_y = int(self.y / (HEIGHT / self.grid_size))
        
        # Use 4 sectors instead of 8
        angle_sector = int(self.angle / 90)
        
        dx = self.beehive_position[0] - self.x
        dy = self.beehive_position[1] - self.y
        beehive_angle = math.degrees(math.atan2(dy, dx)) % 360
        relative_angle = (beehive_angle - self.angle) % 360
        relative_sector = int(relative_angle / 90)
        
        return (grid_x, grid_y, angle_sector, relative_sector)

    def step(self, action):
        prev_distance = self.get_distance_to_beehive()
        prev_x, prev_y = self.x, self.y

        # Execute action
        if action == 'forward':
            dx = math.cos(math.radians(self.angle)) * MOVE_SPEED
            dy = math.sin(math.radians(self.angle)) * MOVE_SPEED
            new_x = self.x + dx
            new_y = self.y + dy
            
            if 0 <= new_x <= WIDTH:
                self.x = new_x
            if 0 <= new_y <= HEIGHT:
                self.y = new_y
                
        elif action == 'rotate_left':
            self.angle = (self.angle - ROTATION_SPEED) % 360
        elif action == 'rotate_right':
            self.angle = (self.angle + ROTATION_SPEED) % 360

        # Calculate rewards
        current_distance = self.get_distance_to_beehive()
        distance_delta = prev_distance - current_distance
        
        # More aggressive reward structure
        reward = 0
        
        # Strong reward for getting closer to beehive
        if distance_delta > 0:
            reward += 5 * (distance_delta / MOVE_SPEED)  # Proportional to improvement
        # else:
        #     reward -= 2  # Stronger penalty for moving away
            
        # Calculate angle to beehive
        dx = self.beehive_position[0] - self.x
        dy = self.beehive_position[1] - self.y
        beehive_angle = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = min(abs(beehive_angle - self.angle), 360 - abs(beehive_angle - self.angle))
        
        # Reward for facing beehive
        if angle_diff < 45:
            reward += 2
            if action == 'forward':
                reward += 3  # Extra reward for moving while facing beehive
        elif angle_diff < 90:
            reward += 1
            
        # Big reward for reaching beehive
        if current_distance < BEEHIVE_SIZE + AGENT_SIZE:
            #reward = 200  # Increased success reward
            reward = (WIDTH-current_distance)*1000
            done = True
        else:
            done = False

        # Strong penalty for hitting boundaries
        # if self.x in [0, WIDTH] or self.y in [0, HEIGHT]:
        #     reward -= 10

        # # Penalty for rotating too much
        # if action in ['rotate_left', 'rotate_right'] and angle_diff < 30:
        #     reward -= 1  # Discourage unnecessary rotation

        return self.get_state(), reward, done

def train_agent():
    # Adjust training parameters
    global TRAINING_EPISODES, MAX_STEPS_PER_EPISODE
    TRAINING_EPISODES = 5000  # More episodes
    MAX_STEPS_PER_EPISODE = 200  # Shorter episodes
    
    print("Training agent...")
    beehive_pos = (WIDTH//2, HEIGHT//2)
    agent = TrainingAgent(beehive_pos)
    episode_rewards = []
    episode_steps = []
    success_rate = []
    window_size = 100

    for episode in tqdm(range(TRAINING_EPISODES)):
        state = agent.reset()
        total_reward = 0
        steps = 0
        
        while steps < MAX_STEPS_PER_EPISODE:
            # Epsilon-greedy action selection
            if random.random() < agent.epsilon:
                action = random.choice(agent.actions)
            else:
                if state not in agent.q_table:
                    agent.q_table[state] = {a: 0.0 for a in agent.actions}
                action = max(agent.q_table[state].items(), key=lambda x: x[1])[0]

            # Take action
            next_state, reward, done = agent.step(action)
            
            # Update Q-table
            if state not in agent.q_table:
                agent.q_table[state] = {a: 0.0 for a in agent.actions}
            if next_state not in agent.q_table:
                agent.q_table[next_state] = {a: 0.0 for a in agent.actions}

            old_value = agent.q_table[state][action]
            next_max = max(agent.q_table[next_state].values())
            new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_value)
            agent.q_table[state][action] = new_value

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                success_rate.append(1)
                break
                
        if steps >= MAX_STEPS_PER_EPISODE:
            success_rate.append(0)

        # Decay epsilon
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Print progress every 100 episodes
        if episode % 1000 == 0:
            recent_success = sum(success_rate[-100:]) / min(100, len(success_rate))
            print(f"Episode {episode}, Success rate: {recent_success:.2f}, Epsilon: {agent.epsilon:.3f}")
            visualize_trained_agent(agent.q_table)

    # Plot training metrics
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(1, 3, 2)
    plt.plot(episode_rewards)
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 3)
    window_success = [sum(success_rate[i:i+window_size])/window_size 
                     for i in range(len(success_rate)-window_size+1)]
    plt.plot(window_success)
    plt.title(f'Success Rate ({window_size}-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    plt.tight_layout()
    plt.show()
    
    return agent.q_table

class VisualAgent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = random.randint(0, 360)
        self.vision_active = True
        self.beehive_position = (x, y)
        self.reward_position = None
        self.carrying_reward = False
        self.path_positions = [(x, y)]
        self.grid_size = GRID_SIZE  # Match training grid size

    def rotate_randomly(self):
        if random.random() < 0.5:
            self.rotate_left()
        else:
            self.rotate_right()

    def move_towards_reward(self):
        if self.reward_position:
            dx = self.reward_position[0] - self.x
            dy = self.reward_position[1] - self.y
            target_angle = math.degrees(math.atan2(dy, dx)) % 360
            
            if abs(target_angle - self.angle) > ROTATION_SPEED:
                if (target_angle - self.angle + 360) % 360 > 180:
                    self.rotate_left()
                else:
                    self.rotate_right()
            else:
                self.move_forward()

    def get_state(self):
        grid_x = int(self.x / (WIDTH / 20))
        grid_y = int(self.y / (HEIGHT / 20))
        angle_sector = int(self.angle / 45)
        dx = self.beehive_position[0] - self.x
        dy = self.beehive_position[1] - self.y
        beehive_angle = math.degrees(math.atan2(dy, dx)) % 360
        beehive_sector = int(beehive_angle / 45)
        return (grid_x, grid_y, angle_sector, beehive_sector)

    def move_forward(self):
        dx = math.cos(math.radians(self.angle)) * MOVE_SPEED
        dy = math.sin(math.radians(self.angle)) * MOVE_SPEED
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x <= WIDTH:
            self.x = new_x
        if 0 <= new_y <= HEIGHT:
            self.y = new_y
        self.path_positions.append((self.x, self.y))

    def rotate_left(self):
        self.angle = (self.angle - ROTATION_SPEED) % 360

    def rotate_right(self):
        self.angle = (self.angle + ROTATION_SPEED) % 360

    def detect_reward(self, reward_pos):
        if self.vision_active and distance((self.x, self.y), reward_pos) <= VISION_RANGE:
            self.reward_position = reward_pos
            print("Reward detected!")
            return True
        return False

    def return_to_beehive(self, q_table):
        state = self.get_state()
        if state in q_table:
            action = max(q_table[state].items(), key=lambda x: x[1])[0]
            print(f"State: {state}, Action: {action}, Q-values: {q_table[state]}")
        else:
            action = random.choice(['forward', 'rotate_left', 'rotate_right'])
            print(f"State {state} not found in Q-table, random action: {action}")
            
        if action == 'forward':
            self.move_forward()
        elif action == 'rotate_left':
            self.rotate_left()
        else:
            self.rotate_right()

    def draw(self, ax):
        agent_circle = patches.Circle(
            (self.x, self.y), 
            AGENT_SIZE, 
            color='red' if self.carrying_reward else 'green'
        )
        ax.add_patch(agent_circle)
        
        if self.vision_active:
            end_x = self.x + VISION_RANGE * math.cos(math.radians(self.angle))
            end_y = self.y + VISION_RANGE * math.sin(math.radians(self.angle))
            ax.plot([self.x, end_x], [self.y, end_y], 'g-')

        if len(self.path_positions) > 1:
            path_x = [p[0] for p in self.path_positions]
            path_y = [p[1] for p in self.path_positions]
            ax.plot(path_x, path_y, 'b-', alpha=0.5)

    def reset_for_new_search(self):
        self.vision_active = True
        self.reward_position = None
        self.carrying_reward = False
        self.path_positions = [(self.x, self.y)]

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def visualize_trained_agent(q_table):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initialize with random beehive position
    beehive = Beehive()
    agent = VisualAgent(beehive.x, beehive.y)
    reward = Reward()
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_aspect('equal')
        
        if agent.vision_active:
            if agent.detect_reward((reward.x, reward.y)):
                agent.vision_active = False
            else:
                agent.rotate_randomly()
                agent.move_forward()
        
        if agent.reward_position and not agent.carrying_reward:
            agent.move_towards_reward()
            if distance((agent.x, agent.y), (reward.x, reward.y)) <= AGENT_SIZE + REWARD_SIZE:
                agent.carrying_reward = True
                print("Reward picked up!")

        if agent.carrying_reward:
            agent.return_to_beehive(q_table)
            if distance((agent.x, agent.y), (beehive.x, beehive.y)) <= AGENT_SIZE + BEEHIVE_SIZE:
                print("Reward delivered!")
                agent.reset_for_new_search()
                reward.reposition()

        agent.draw(ax)
        beehive.draw(ax)
        reward.draw(ax)
        
        # Show Q-table size
        ax.text(10, HEIGHT - 20, f'Q-table size: {len(q_table)}', fontsize=10)

    ani = FuncAnimation(fig, update, frames=None, interval=2, cache_frame_data=False)
    plt.show()

class Beehive:
    def __init__(self):
        self.x = random.randint(BEEHIVE_SIZE, WIDTH - BEEHIVE_SIZE)
        self.y = random.randint(BEEHIVE_SIZE, HEIGHT - BEEHIVE_SIZE)

    def draw(self, ax):
        beehive_rect = patches.Rectangle(
            (self.x - BEEHIVE_SIZE // 2, self.y - BEEHIVE_SIZE // 2),
            BEEHIVE_SIZE, BEEHIVE_SIZE, color='brown'
        )
        ax.add_patch(beehive_rect)

class Reward:
    def __init__(self):
        self.reposition()

    def reposition(self):
        self.x = random.randint(REWARD_SIZE, WIDTH - REWARD_SIZE)
        self.y = random.randint(REWARD_SIZE, HEIGHT - REWARD_SIZE)

    def draw(self, ax):
        reward_circle = patches.Circle((self.x, self.y), REWARD_SIZE, color='red')
        ax.add_patch(reward_circle)

def main():
    print("Training agent...")
    q_table = train_agent()
    
    print("\nTraining complete! Starting visualization...")
    visualize_trained_agent(q_table)

if __name__ == "__main__":
    main()