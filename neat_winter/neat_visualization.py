from manim import *
import neat
import numpy as np
from agent import Agent, distance
from constants import *
import math
import pickle
import os


class NetworkVisualization(Scene):
    def construct(self):
        # Initial setup
        self.camera.frame_width = 20
        self.camera.frame_height = 10
        
        # Load genome and network
        with open("runs/run_20241229_004508/genomes/champion_genome_gen1600.pkl", 'rb') as f:
            data = pickle.load(f)
            genome = data['genome']
            config = data['config']
        
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Part 1: Show full network
        title = Text("Neural Network Architecture", font_size=72)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        network_group = self.create_network_viz(genome, config)
        network_group.scale(1.5)  # Make it bigger
        network_group.next_to(title, DOWN)
        
        # Show connections first
        self.play(
            *[Create(conn) for conn in network_group[0]], 
            run_time=2,
            lag_ratio=0.01
        )
        
        # Then nodes with labels (combine animations)
        nodes = network_group[1]
        labels = network_group[2]
        self.play(
            *[Create(node) for node in nodes],
            *[Write(label) for label in labels],
            run_time=2,
        )
        self.wait()

        # Part 2: Finding the compass node
        compass_node = self.find_compass_node(network, genome, config)
        compass_info = Text("Finding compass node...", font_size=36)
        compass_info.next_to(network_group, DOWN)
        self.play(Write(compass_info))

        # Just highlight a few sample nodes instead of all
        sample_nodes = [nodes[i] for i in range(0, len(nodes), len(nodes)//5)]  # Show 5 sample nodes
        for node in sample_nodes:
            highlight = Circle(radius=node.radius + 0.1)
            highlight.set_stroke(YELLOW, opacity=0.5)
            highlight.move_to(node)
            self.play(
                FadeIn(highlight),
                FadeOut(highlight),
                run_time=0.2
            )

        # Highlight compass node
        compass_circle = nodes[self.get_node_index(compass_node, network_group)]
        compass_highlight = Circle(radius=compass_circle.radius + 0.1)
        compass_highlight.set_stroke(YELLOW, 2)
        compass_highlight.move_to(compass_circle)
        
        compass_found_text = Text("Found compass node!", color=YELLOW, font_size=36)
        compass_found_text.next_to(compass_info, DOWN)
        
        self.play(
            ReplacementTransform(compass_info, compass_found_text),
            Create(compass_highlight)
        )
        self.wait()

        # Part 3: Show input connections
        input_nodes, input_connections = self.get_compass_inputs(genome, compass_node, network_group)
        
        input_text = Text("Input connections to compass node", font_size=36)
        input_text.next_to(compass_found_text, DOWN)
        
        # Highlight input connections
        for conn in input_connections:
            conn_highlight = conn.copy()
            conn_highlight.set_stroke(YELLOW, width=4)
            self.play(
                Create(conn_highlight),
                run_time=0.5
            )
        self.wait()

        # Part 4: Zoom into relevant parts
        relevant_group = VGroup()
        relevant_group.add(compass_highlight)
        relevant_group.add(*[node.copy() for node in input_nodes])
        relevant_group.add(*[conn.copy() for conn in input_connections])
        
        # Fade out irrelevant parts
        self.play(
            *[FadeOut(obj) for obj in network_group if obj not in relevant_group],
            run_time=2
        )
        
        # Move and scale relevant parts
        self.play(
            relevant_group.animate.scale(1.2).move_to(LEFT * 3),
            run_time=2
        )

        # Part 5: Show agent behavior
        env_title = Text("Agent Behavior", font_size=48)
        env_title.next_to(relevant_group, RIGHT).shift(UP)
        
        env = self.create_environment()
        env.next_to(env_title, DOWN)
        
        self.play(
            Write(env_title),
            Create(env)
        )

        # Create activation plot
        activation_graph = self.create_activation_graph()
        activation_graph.next_to(env, DOWN)
        self.play(Create(activation_graph))

        # Show agent behavior
        self.animate_agent_behavior(env, activation_graph, network)
        
    def create_network_viz(self, genome, config):
        """Creates the neural network visualization"""
        group = VGroup()
        
        # Get node sets
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        hidden_nodes = [n for n in genome.nodes.keys() 
                       if n not in input_nodes and n not in output_nodes]
        
        # Create layered layout
        positions = {}
        layers = [input_nodes, hidden_nodes, output_nodes]
        layer_spacing = 4  # Horizontal space between layers
        node_spacing = 1.5  # Vertical space between nodes
        
        for i, layer in enumerate(layers):
            layer_size = len(layer)
            for j, node in enumerate(layer):
                x = layer_spacing * (i - 1)  # Center the network horizontally
                y = node_spacing * (j - layer_size/2)  # Center the layer vertically
                positions[node] = np.array([x, y, 0])

        # Draw connections first (behind nodes)
        connections = VGroup()
        for conn in genome.connections.values():
            if conn.enabled:
                start = positions[conn.key[0]]
                end = positions[conn.key[1]]
                
                # Color based on weight
                color = BLUE_B if conn.weight > 0 else RED_B
                opacity = min(abs(conn.weight), 1.0)
                width = 2 * abs(conn.weight)
                
                line = Line(
                    start=start,
                    end=end,
                    stroke_width=width,
                    stroke_opacity=opacity
                )
                line.set_stroke(color)
                connections.add(line)
        
        # Draw nodes
        nodes = VGroup()
        labels = VGroup()
        
        for node_id, pos in positions.items():
            # Determine node properties based on type
            if node_id in input_nodes:
                color = BLUE
                label = f"In {node_id}"
                radius = 0.2
            elif node_id in output_nodes:
                color = GREEN
                label = f"Out {node_id}"
                radius = 0.2
            else:
                color = YELLOW
                label = f"H {node_id}"
                radius = 0.15
            
            # Create node circle
            circle = Circle(radius=radius)
            circle.set_stroke(color, 2)
            circle.set_fill(BLACK, 1)
            circle.move_to(pos)
            circle.node_id = node_id  # Store ID for later reference
            
            # Create text label
            text = Text(label, font_size=24)
            text.next_to(circle, DOWN, buff=0.1)
            text.set_fill(color)
            
            nodes.add(circle)
            labels.add(text)
        
        group.add(connections, nodes, labels)
        return group

    def find_compass_node(self, network, genome, config):
        """Analyzes network to find the compass node"""
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        hidden_nodes = [n for n in genome.nodes.keys() 
                       if n not in input_nodes
                       and n not in output_nodes]
        
        max_sinusoidal = 0
        compass_node = None
        
        # Test each hidden node's response to rotation
        angles = np.linspace(0, 360, 360)
        for node in hidden_nodes:
            activations = []
            for angle in angles:
                # Create input vector
                heading_x = np.cos(np.radians(angle))
                heading_y = np.sin(np.radians(angle))
                inputs = [heading_x, heading_y] + [0] * 15  # Rest zeroed
                
                # Get outputs after activation
                outputs = network.activate(inputs)

                # Get node activation by examining node's output
                activation = genome.nodes[node].response
                activations.append(activation)
            
            # Check if activation pattern is sinusoidal
            fft = np.fft.fft(activations)
            main_freq = np.abs(fft[1]) / len(fft)
            if main_freq > max_sinusoidal:
                max_sinusoidal = main_freq
                compass_node = node
                
        return compass_node

    def get_node_index(self, node_id, network_group):
        """Helper to find node in visualization group"""
        nodes = network_group[1]  # Get the nodes group
        for i, node in enumerate(nodes):
            if hasattr(node, 'node_id') and node.node_id == node_id:
                return i
        print(f"Warning: Could not find node {node_id}")  # Debug print
        return 0  # Return first node as fallback instead of None

    def get_compass_inputs(self, genome, compass_node, network_group):
        """Find input nodes and connections that feed into the compass node"""
        input_node_ids = []
        input_connections = []

        # Find connections feeding into compass node
        for conn in genome.connections.values():
            if conn.enabled and conn.key[1] == compass_node:
                input_node_ids.append(conn.key[0])
                
                # Find corresponding connection visualization
                for line in network_group[0]:  # Connections are first group
                    # Check if this line represents this connection
                    if np.allclose(line.start.get_center(), network_group[1][self.get_node_index(conn.key[0], network_group)].get_center()) and \
                       np.allclose(line.end.get_center(), network_group[1][self.get_node_index(conn.key[1], network_group)].get_center()):
                        input_connections.append(line)
                        break
        
        # Find corresponding node objects
        input_nodes = []
        for node_id in input_node_ids:
            node_index = self.get_node_index(node_id, network_group)
            if node_index is not None:
                input_nodes.append(network_group[1][node_index])
        
        return input_nodes, input_connections

    def create_environment(self):
        """Creates the agent's environment visualization"""
        env = VGroup()
        
        # Boundary (match your WIDTH/HEIGHT constants)
        boundary = Rectangle(width=WIDTH/25, height=HEIGHT/25)
        boundary.set_stroke(WHITE, 1)
        
        # Beehive at center
        beehive = Square(side_length=BEEHIVE_SIZE/25)
        beehive.set_fill(YELLOW, 0.5)
        beehive.set_stroke(YELLOW_E, 2)
        beehive.move_to(ORIGIN)
        
        # Reward/Pollen at a typical distance
        reward = Circle(radius=REWARD_SIZE/25)
        reward.set_fill(RED, 0.5)
        reward.set_stroke(RED_E, 2)
        # Position at typical distance from hive
        reward.move_to(2 * RIGHT + UP)  
        
        # Agent with vision cone
        agent = Dot(color=GREEN)
        vision_cone = AnnularSector(
            inner_radius=0,
            outer_radius=VISION_RANGE/25,  # Scale down from your constants
            angle=VISION_CONE_ANGLE * DEGREES,
            start_angle=-VISION_CONE_ANGLE * DEGREES / 2,
            color=GREEN,
            fill_opacity=0.2
        )
        vision_cone.move_to(agent)
        
        # Add path tracker
        path = VMobject(stroke_color=BLUE_E, stroke_width=1)
        path.set_points_as_corners([agent.get_center()])
        
        env.add(boundary, beehive, reward, vision_cone, agent, path)
        return env

    def create_activation_graph(self):
        """Creates a graph for showing compass neuron activation"""
        axes = Axes(
            x_range=[-3, 3, 1],  # x-axis from -3 to 3
            y_range=[-1.5, 1.5, 0.5],  # y-axis from -1.5 to 1.5
            axis_config={"color": WHITE}
        )

        # Add degree markers
        x_ticks = VGroup()
        for angle in [0, 90, 180, 270, 360]:
            x = 3 * (angle / 180 - 1)
            tick = Line(0.1 * UP, 0.1 * DOWN).move_to(axes.c2p(x, 0))
            x_ticks.add(tick)

        y_ticks = VGroup()
        for val in [-1, 0, 1]:
            tick = Line(0.1 * LEFT, 0.1 * RIGHT).move_to(axes.c2p(0, val))
            y_ticks.add(tick)

        axes.add(x_ticks, y_ticks)

        # Add simple text labels
        x_label = Text("Angle", font_size=24).next_to(axes, DOWN)
        y_label = Text("Activation", font_size=24).next_to(axes, LEFT)
        axes.add(x_label, y_label)

        return axes


    def angle_to_graph_coords(self, angle, activation, graph):
        """Convert angle and activation to graph coordinates"""
        x = 3 * (angle/180 - 1)  # Map 0-360 to -3 to 3
        y = 1.5 * activation  # Map -1 to 1 to -1.5 to 1.5
        return graph[0].c2p(x, y)  # Convert to scene coordinates using x-axis

    def animate_agent_behavior(self, env, activation_graph, network):
        """Shows complete agent behavior cycle with network activations"""
        
        # Extract components
        boundary = env[0]
        beehive = env[1]
        reward = env[2]
        vision_cone = env[3]
        agent = env[4]

        # Track activation with properly initialized path
        activation_dot = Dot(color=YELLOW)
        activation_dot.move_to(activation_graph[0].get_center())  # Start at origin
        activation_path = VMobject(stroke_color=YELLOW)
        activation_path.set_points([activation_dot.get_center()])  # Initialize with start point
        
        # Define update function for path
        def update_path(path):
            points = path.get_points()
            points = np.vstack([points, [activation_dot.get_center()]])
            path.set_points(points)
        
        # Add updater AFTER initializing points
        activation_path.add_updater(update_path)
        self.add(activation_dot, activation_path)

        # Phase 1: Initial search with vision
        search_text = Text("Phase 1: Searching with vision", font_size=36)
        search_text.to_edge(UP)
        self.play(Write(search_text))
        
        # Move agent around with vision cone
        vision_cone.set_fill(opacity=0.3)
        initial_path = [
            RIGHT + UP,
            RIGHT + DOWN,
            LEFT + DOWN,
            LEFT + UP,
        ]
        
        for target in initial_path:
            self.play(
                agent.animate.move_to(target),
                vision_cone.animate.move_to(target),
                run_time=2
            )
            # Update activation plot
            angle = np.degrees(np.arctan2(target[1], target[0]))
            x = 3 * (angle/180)
            y = np.sin(np.radians(angle))
            self.play(
                activation_dot.animate.move_to(activation_graph.c2p(x, y)),
                run_time=0.5
            )

        # Phase 2: Found pollen
        found_text = Text("Phase 2: Found pollen!", font_size=36)
        found_text.to_edge(UP)
        self.play(
            FadeOut(search_text),
            FadeIn(found_text)
        )
        
        self.play(
            agent.animate.move_to(reward.get_center()),
            vision_cone.animate.move_to(reward.get_center())
        )
        
        # Phase 3: Vision turned off
        blind_text = Text("Phase 3: Vision turned off, using compass", font_size=36)
        blind_text.to_edge(UP)
        self.play(
            FadeOut(found_text),
            FadeIn(blind_text),
            FadeOut(vision_cone),
            agent.animate.set_color(RED)
        )
        
        # Return journey using compass
        return_path = [
            2 * LEFT + UP,
            2 * LEFT,
            LEFT,
            ORIGIN
        ]
        
        for target in return_path:
            self.play(
                agent.animate.move_to(target),
                run_time=2
            )
            # Update activation plot
            angle = np.degrees(np.arctan2(target[1], target[0]))
            x = 3 * (angle/180)
            y = np.sin(np.radians(angle))
            self.play(
                activation_dot.animate.move_to(activation_graph.c2p(x, y)),
                run_time=0.5
            )
        
        # Success!
        success_text = Text("Successfully returned home!", color=GREEN, font_size=36)
        success_text.to_edge(UP)
        self.play(
            FadeOut(blind_text),
            FadeIn(success_text),
            agent.animate.set_color(GREEN)
        )
        self.wait(2)


if __name__ == "__main__":
    # Command to render:
    # manim -pql neat_visualization.py NetworkVisualization
    pass