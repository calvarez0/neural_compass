# NEAT NEURAL COMPASS

A Python implementation of neuroevolution using NEAT (NeuroEvolution of Augmenting Topologies) to train agents for foraging behavior. The agents learn to navigate their environment, locate rewards, and return to their home base (similar to a bee leaving the hive and returning with pollen).

These are early research experiments inspired by Dr. Gaby Maimon's Drosophila neural circuitry for navigation. Although I can't share all my business or research code, I strongly believe in openly sharing as much code as possible for research rather than falling into the common trap of defensiveness or gatekeeping seen among researchers.

The rest of this readme is AI-generated.

![Agent Simulation](simulation.gif)

## Features

- Advanced agent navigation with vision cone system
- Spatial memory grid for environment tracking
- Energy management system
- Path integration for movement optimization
- Comprehensive visualization tools for network topology and agent behavior
- Automated checkpoint system for training progress

## Requirements

```bash
pip install neat-python numpy matplotlib graphviz
```

## Main Project Structure

- `main.py` - Core training loop and NEAT implementation
- `agent.py` - Agent class with navigation and sensor systems
- `constants.py` - Global configuration constants
- `utilities.py` - Helper functions for visualization and data management
- `activation_test.py` - Interactive visualization of trained agents

## Usage

### Training New Agents

```bash
python main.py
```

Training progress is automatically saved in timestamped directories under `runs/`. Each run includes:
- Network topology visualizations
- Fitness statistics
- Agent simulation videos
- Saved genomes

### Testing Trained Agents

```bash
python activation_test.py
```

Modify the genome path in `activation_test.py` to load different trained agents:

```python
genome_path = "runs/run_TIMESTAMP/genomes/champion_genome_genXXX.pkl"
```

## Agent Behavior

Agents are trained to:
1. Explore efficiently using vision and spatial memory
2. Locate and collect rewards
3. Navigate back to home base without vision
4. Optimize paths for energy efficiency

Energy costs are associated with:
- Movement (scaled with speed)
- Rotation (higher when carrying reward)
- Base operation

## Configuration

Key parameters in `constants.py`:
- Environment dimensions: `WIDTH`, `HEIGHT`
- Vision parameters: `VISION_RANGE`, `VISION_CONE_ANGLE`
- Movement parameters: `ROTATION_SPEED`, `MOVE_SPEED`
- Memory system: `GRID_SIZE`, `MEMORY_DECAY`

NEAT-specific parameters are in `config-neat`.