import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Constants
WIDTH, HEIGHT = 800, 800
AGENT_SIZE = 10
REWARD_SIZE = 10
BEEHIVE_SIZE = 20
VISION_RANGE = 250
ROTATION_SPEED = 10
MOVE_SPEED = 5

# RL parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2
GRID_SIZE = 20

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = random.randint(0, 360)
        self.vision_active = True
        self.beehive_position = (x, y)
        self.reward_position = None
        self.carrying_reward = False
        self.path_positions = [(x, y)]
        
        # RL components
        self.q_table = {}
        self.actions = ['forward', 'rotate_left', 'rotate_right']
        self.previous_state = None
        self.previous_action = None

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
        grid_x = int(self.x / (WIDTH / GRID_SIZE))
        grid_y = int(self.y / (HEIGHT / GRID_SIZE))
        angle_sector = int(self.angle / 45)
        
        dx = self.beehive_position[0] - self.x
        dy = self.beehive_position[1] - self.y
        beehive_angle = math.degrees(math.atan2(dy, dx)) % 360
        beehive_sector = int(beehive_angle / 45)
        
        return (grid_x, grid_y, angle_sector, beehive_sector)

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state][action]

    def choose_action(self):
        current_state = self.get_state()
        
        if random.random() < EPSILON:
            return random.choice(self.actions)
        else:
            if current_state not in self.q_table:
                self.q_table[current_state] = {a: 0.0 for a in self.actions}
            return max(self.q_table[current_state].items(), key=lambda x: x[1])[0]

    def update_q_value(self, reward):
        if self.previous_state is not None and self.previous_action is not None:
            current_state = self.get_state()
            
            old_value = self.get_q_value(self.previous_state, self.previous_action)
            next_max = max(self.get_q_value(current_state, a) for a in self.actions)
            
            new_value = old_value + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * next_max - old_value
            )
            
            if self.previous_state not in self.q_table:
                self.q_table[self.previous_state] = {a: 0.0 for a in self.actions}
            self.q_table[self.previous_state][self.previous_action] = new_value

    def draw(self, ax):
        agent_circle = patches.Circle((self.x, self.y), AGENT_SIZE, 
                                    color='red' if self.carrying_reward else 'green')
        ax.add_patch(agent_circle)
        
        if self.vision_active:
            end_x = self.x + VISION_RANGE * math.cos(math.radians(self.angle))
            end_y = self.y + VISION_RANGE * math.sin(math.radians(self.angle))
            ax.plot([self.x, end_x], [self.y, end_y], 'g-')

        if len(self.path_positions) > 1:
            path_x = [p[0] for p in self.path_positions]
            path_y = [p[1] for p in self.path_positions]
            ax.plot(path_x, path_y, 'b-', alpha=0.5)

    def execute_action(self, action):
        self.previous_state = self.get_state()
        self.previous_action = action
        
        if action == 'forward':
            self.move_forward()
        elif action == 'rotate_left':
            self.rotate_left()
        elif action == 'rotate_right':
            self.rotate_right()
            
        distance_to_beehive = distance((self.x, self.y), self.beehive_position)
        
        if distance_to_beehive < BEEHIVE_SIZE + AGENT_SIZE:
            reward = 100
        else:
            reward = -0.1 - (distance_to_beehive / WIDTH)
            
        self.update_q_value(reward)
        return reward

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
            return True
        return False

    def return_to_beehive(self):
        action = self.choose_action()
        reward = self.execute_action(action)
        return reward

    def reset_for_new_search(self):
        self.vision_active = True
        self.reward_position = None
        self.carrying_reward = False
        self.path_positions = [(self.x, self.y)]
        self.previous_state = None
        self.previous_action = None

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

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def main():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    beehive = Beehive()
    agent = Agent(beehive.x, beehive.y)
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

        if agent.carrying_reward:
            rl_reward = agent.return_to_beehive()
            if distance((agent.x, agent.y), (beehive.x, beehive.y)) <= AGENT_SIZE + BEEHIVE_SIZE:
                agent.carrying_reward = False
                agent.reset_for_new_search()
                reward.reposition()

        agent.draw(ax)
        beehive.draw(ax)
        reward.draw(ax)
        
        ax.text(10, HEIGHT - 20, f'Q-table size: {len(agent.q_table)}', fontsize=10)

    ani = FuncAnimation(fig, update, frames=None, interval=1, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()