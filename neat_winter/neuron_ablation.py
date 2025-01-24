import neat
import pickle
from agent import Agent
from utilities import randomize_reward_position
from constants import WIDTH, HEIGHT, REWARD_SIZE


def ablate_neuron(genome, neuron_id, backup_connections):
    """
    Temporarily disable a neuron by removing its connections, restoring them afterward.
    """
    for conn_key in list(genome.connections.keys()):
        if conn_key[0] == neuron_id or conn_key[1] == neuron_id:
            backup_connections[conn_key] = genome.connections[conn_key]
            del genome.connections[conn_key]


def restore_connections(genome, backup_connections):
    """
    Restore connections from the backup after ablation testing.
    """
    genome.connections.update(backup_connections)


def analyze_node_dependencies(genome, config):
    """
    Identify node dependencies by counting connections to outputs.
    """
    output_nodes = config.genome_config.output_keys
    hidden_nodes = [
        node_id for node_id in genome.nodes.keys()
        if node_id not in config.genome_config.input_keys + output_nodes
    ]
    
    node_dependencies = {node_id: 0 for node_id in hidden_nodes}
    for conn in genome.connections.values():
        if conn.enabled and conn.key[1] in output_nodes:
            if conn.key[0] in hidden_nodes:
                node_dependencies[conn.key[0]] += 1
    
    return node_dependencies


def test_network_performance(genome_path, test_function):
    """
    Load genome, ablate neurons, and test performance.
    """
    with open(genome_path, 'rb') as f:
        data = pickle.load(f)
    genome, config = data['genome'], data['config']
    node_dependencies = analyze_node_dependencies(genome, config)
    sorted_nodes = sorted(node_dependencies.keys(), key=lambda x: -node_dependencies[x])

    for neuron_id in sorted_nodes:
        print(f"Testing ablation for neuron {neuron_id}...")
        backup_connections = {}
        ablate_neuron(genome, neuron_id, backup_connections)

        try:
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            performance = test_function(network)
        except Exception as e:
            print(f"Error during ablation: {e}")
            performance = float('inf')

        print(f"Performance after ablating neuron {neuron_id}: {performance}")
        restore_connections(genome, backup_connections)


def test_function(network):
    """
    Simulate the agent's performance with dynamic inputs to evaluate network performance.
    """
    agent = Agent((WIDTH / 2, HEIGHT / 2))
    reward_pos = randomize_reward_position(WIDTH, HEIGHT, REWARD_SIZE, agent.beehive_position)
    total_steps = 0

    for _ in range(100):
        inputs = agent.get_sensor_data(reward_pos)
        outputs = network.activate(inputs)
        agent.update(outputs, reward_pos)
        total_steps += 1

        if agent.returned_home:
            return total_steps  # Task completed successfully

    # Partial completion scoring
    if agent.carrying_reward:
        return total_steps + 50  # Found the reward but didn't return
    elif agent.found_reward:
        return total_steps + 100  # Found the reward but didn't pick it up

    return float('inf')  # Failed to achieve any task


if __name__ == "__main__":
    test_network_performance('runs/run_20241229_004508/genomes/champion_genome_gen2000.pkl', test_function)
