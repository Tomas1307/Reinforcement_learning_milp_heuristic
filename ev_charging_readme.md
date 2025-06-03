# Multi-Agent Evolution for Electric Vehicle Charging Systems: A Scatter Search Approach to Breeding Specialized DQN Agents

A comprehensive electric vehicle charging management and optimization system that combines Deep Q-Network (DQN) Reinforcement Learning, Multi-Objective MILP Optimization, and Scatter Search metaheuristics to find optimal hyperparameter configurations for intelligent charging stations.

**Author:** Tomás Acosta Bernal  
**Institution:** Universidad de los Andes  
**Year:** 2025

## Overview

This system addresses the complex challenge of Electric Vehicle (EV) charging scheduling in smart grid environments, drawing from recent advances in multi-agent reinforcement learning, optimal scheduling strategies with V2G integration, and robust scheduling under uncertainty. The implementation combines distributed EV scheduling approaches with heuristic optimization methods and intelligent energy management algorithms.

The core innovation lies in the integration of Scatter Search metaheuristics with Deep Q-Network reinforcement learning for automated hyperparameter optimization, creating a robust system capable of adapting to diverse operational scenarios and customer priorities.

## Key Features

- **Reinforcement Learning Agent**: Enhanced DQN with Dueling architecture and automatic GPU support
- **Multi-Objective MILP Optimization**: Epsilon-constraint solver balancing cost minimization and customer satisfaction
- **Scatter Search Metaheuristic**: Automatic hyperparameter optimization with archetype classification
- **Advanced Analytics**: Comprehensive visualizations and detailed convergence reports
- **Multi-System Evaluation**: Automated testing across multiple charging station configurations
- **Hybrid RL+MILP**: Combined approach leveraging strengths of both methodologies

## Project Structure

```
DQN_AGENTS/
├── src/
│   ├── common/                     # Shared utilities and core functions
│   │   ├── config.py              # Configuration loading and system management
│   │   ├── logger.py              # Episode logging and training metrics
│   │   ├── metrics.py             # Performance evaluation metrics
│   │   ├── utils.py               # General utilities and helpers
│   │   └── visualize.py           # Gantt charts and visualization tools
│   ├── configs/                   # System and algorithm configurations
│   │   ├── system_data/           # Test system configurations (JSON)
│   │   ├── hyperparameter_ranges.yaml    # Scatter Search parameter ranges
│   │   ├── hyperparameters.yaml          # Base DQN hyperparameters
│   │   └── scatter_search_config.yaml    # Algorithm configuration
│   ├── dqn_agent/                # Deep Q-Network Reinforcement Learning
│   │   ├── agent.py              # Enhanced DQN with PyTorch and Dueling
│   │   ├── environment.py        # Optimized charging environment simulation
│   │   └── training.py           # Multi-system training functions
│   ├── milp_optimizer/           # Multi-Objective Linear Programming
│   │   └── optimizer.py          # PuLP-based epsilon-constraint solver
│   ├── scatter_search/           # Metaheuristic Hyperparameter Optimization
│   │   ├── scatter_algorithm.py          # Main algorithm with checkpointing
│   │   ├── hyperparameter_space.py       # Search space and archetypes
│   │   ├── cosine_similarity_hyperparams.py  # Diversity metrics
│   │   ├── results_analyzer.py           # Analysis and export tools
│   │   └── advanced_analysis.py          # Convergence visualizations
│   └── tests/                    # Evaluation and testing modules
│       └── evaluator.py          # Efficient slot-by-slot simulator
├── model/                        # Trained models (.pt files)
├── results/                      # Optimization results and outputs
│   ├── scatter_search/           # Scatter Search optimization results
│   ├── MILP_RL/                 # Hybrid optimizations
│   └── dqn_agent_system_*/      # Individual system training results
├── requirements.txt             # Python dependencies
├── main.py                      # Main execution script with 8 modes
└── README.md                    # This documentation
```

## Installation and Setup

### System Requirements

- Python 3.11+
- CUDA 12.6+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB recommended for Scatter Search)

### Dependencies Installation

```bash
# Install from requirements file
pip install -r requirements.txt

# Verify PyTorch installation with CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Initial Setup

```bash
# Create necessary directories
mkdir -p results model checkpoints
mkdir -p results/scatter_search/{checkpoints,configurations,visualizations}

# Verify test system data
ls src/configs/system_data/test_system_*.json
```

## Operating Modes

The system provides 8 comprehensive modes accessible through `main.py`:

### 1. Individual DQN Training (`train_dqn`)

Train DQN agents on specific charging station configurations:

```bash
# Train on specific system
python -m src.main --mode train_dqn --system_id 1 --num_episodes 1000

# Train on all available systems
python -m src.main --mode train_dqn --all_systems --num_episodes 500

# Custom hyperparameters training
python -m src.main --mode train_dqn --system_id 1 \
  --hyperparameters_path ./src/configs/hyperparameters.yaml \
  --num_episodes 1000 --log_frequency 50
```

**Generated outputs:**
```
results/dqn_agent_system_1/
├── models/ev_scheduler_model_pytorch_system_1.pt
├── logs/dqn_episode_log_system_1.json
└── plots/dqn_learning_progress.png
```

### 2. Multi-System Training (`train`)

Sequential training across multiple systems with automatic checkpointing:

```bash
python -m src.main --mode train --episodes 30 \
  --model_path ./model/ev_scheduler_model_pytorch.pt
```

### 3. Scatter Search Optimization (`scatter_search`)

Primary optimization mode that automatically discovers optimal hyperparameter configurations:

```bash
# Complete optimization (8-30 hours typical)
python -m src.main --mode scatter_search \
  --scatter_config src/configs/scatter_search_config.yaml \
  --output_base_dir ./results/scatter_search

# Automatic resume from checkpoints
python -m src.main --mode scatter_search  # Prompts for resume option
```

**Algorithm Configuration (`scatter_search_config.yaml`):**
```yaml
algorithm:
  population_size: 200          # Initial population size
  ref_set_size: 20             # Reference set size
  elite_count: 12              # Elite solutions
  diverse_count: 8             # Diverse solutions
  max_iterations: 50           # Maximum iterations
  max_time_hours: 30           # Time limit

evaluation:
  fast:    {systems: [1,2,3], episodes: 30}               # Initial filtering
  medium:  {systems: [1,2,3,4,5,6,7,8,9], episodes: 60}  # Intermediate evaluation
  full:    {systems: [1-17], episodes: 80}                # Complete evaluation
```

**Generated Results:**
```
results/scatter_search/
├── configurations/              # Optimized YAML configurations
│   ├── cost_minimizer_rank_1.yaml
│   ├── satisfaction_maximizer_rank_1.yaml
│   ├── balanced_optimizer_rank_1.yaml
│   ├── urgency_focused_rank_1.yaml
│   └── efficiency_focused_rank_1.yaml
├── trained_models/             # Trained models (.pt files)
│   ├── efficiency_focused_rank_1.pt
│   └── urgency_focused_rank_1.pt
├── visualizations/             # Convergence plots and analysis
├── checkpoints/               # Iteration checkpoints (every 3 iterations)
├── optimization_report.txt    # Executive summary report
└── deployment_package/        # Production-ready package
```

### 4. Solution Evaluation (`solution_scatter_search`)

Evaluate optimized models on test systems:

```bash
# Evaluate specific model on single system
python -m src.main --mode solution_scatter_search \
  --model_to_evaluate ./results/scatter_search/trained_models/efficiency_focused_rank_1.pt \
  --model_name efficiency_focused_rank_1 \
  --system_id 1

# Evaluate across all systems
python -m src.main --mode solution_scatter_search \
  --model_to_evaluate ./results/scatter_search/trained_models/balanced_optimizer_rank_1.pt \
  --model_name balanced_optimizer_rank_1 \
  --all_systems
```

**Per-system output:**
```
results/scatter_search/model_solution/efficiency_focused_rank_1/
├── config_1.json              # Detailed metrics for system 1
├── config_2.json              # Detailed metrics for system 2
└── config_N.json              # ... for each system
```

### 5. RL Solution Generation (`solve`)

Generate solutions using existing DQN models:

```bash
# Single system solution
python -m src.main --mode solve --system_id 1 \
  --model_path ./model/ev_scheduler_model_pytorch.pt

# All systems solutions
python -m src.main --mode solve --all_systems
```

### 6. Hybrid RL+MILP Optimization (`optimize`)

Combine Scatter Search solutions with MILP refinement:

```bash
# Optimize specific system with hybrid approach
python -m src.main --mode optimize \
  --model_name efficiency_focused_rank_1 \
  --system_id 1 \
  --alpha_cost 0.6 --alpha_satisfaction 0.4 \
  --time_limit 900

# Optimize all systems
python -m src.main --mode optimize \
  --model_name balanced_optimizer_rank_1 \
  --all_systems --time_limit 600
```

**Optimization parameters:**
- `--alpha_cost`: Cost objective weight (0.0-1.0)
- `--alpha_satisfaction`: Satisfaction objective weight (0.0-1.0)
- `--time_limit`: MILP solver time limit (seconds)

### 7. Pure MILP Optimization (`run_milp`)

Direct MILP optimization without RL:

```bash
python -m src.main --mode run_milp --system_id 1 --time_limit 300
```

### 8. Solution Visualization (`visualize_solution`)

Generate Gantt chart visualizations of charging schedules:

```bash
python -m src.main --mode visualize_solution \
  --solution_to_visualize ./results/rl_solution_1.json \
  --system_id 1
```

## Behavioral Archetypes

The system automatically identifies 5 distinct behavioral archetypes during Scatter Search optimization, each optimized for specific operational contexts:

### Cost Minimizer
- **Characteristics**: `energy_cost_weight ≥ 1.0`, `cost_ratio ≥ 0.4`
- **Optimal for**: Public charging stations, cost-sensitive operations

```bash
python -m src.main --mode solution_scatter_search \
  --model_name cost_minimizer_rank_1 --system_id 1
```

### Satisfaction Maximizer
- **Characteristics**: `energy_satisfaction_weight ≥ 3.5`, `energy_cost_weight < 0.5`
- **Optimal for**: Premium hotels, luxury shopping centers, executive facilities

```bash
python -m src.main --mode solution_scatter_search \
  --model_name satisfaction_maximizer_rank_1 --system_id 1
```

### Balanced Optimizer
- **Characteristics**: Optimal cost-satisfaction equilibrium
- **Optimal for**: Corporate offices, general-purpose facilities

```bash
python -m src.main --mode solution_scatter_search \
  --model_name balanced_optimizer_rank_1 --system_id 1
```

### Urgency Focused
- **Characteristics**: `penalty_skipped_vehicle > 200.0`
- **Optimal for**: Hospitals, airports, emergency services

```bash
python -m src.main --mode solution_scatter_search \
  --model_name urgency_focused_rank_1 --system_id 1
```

### Efficiency Focused
- **Characteristics**: `reward_assigned_vehicle > 100.0`
- **Optimal for**: Commercial fleets, high-throughput transit stations

```bash
python -m src.main --mode solution_scatter_search \
  --model_name efficiency_focused_rank_1 --system_id 1
```

## Results Analysis and Interpretation

### Automatic Post-Optimization Analysis

```bash
# Generate comprehensive convergence analysis
python -m src.scatter_search.advanced_analysis --base_dir ./results/scatter_search
```

**Generated visualizations:**
- `convergence_analysis.png`: Fitness evolution, diversity, and timing
- `archetype_evolution.png`: Archetype distribution over iterations
- `hyperparameter_evolution.png`: Key hyperparameter evolution
- `diversity_vs_performance.png`: Diversity-performance relationship
- `final_solutions_analysis.png`: Final solution characteristics

### Performance Metrics

**Energy Performance:**
- Total Satisfaction (%): `energy_delivered / energy_required`
- Energy Deficit (kWh): Unmet energy demand
- Efficiency per Vehicle: Average kWh/vehicle

**Economic Performance:**
- Total Cost ($): Dynamic pricing-based cost
- Cost per kWh ($/kWh): Economic efficiency
- Savings vs Baseline (%): Improvement over simple strategies

**Operational Performance:**
- Assignment Ratio (%): `vehicles_served / total_vehicles`
- Resource Utilization (%): Charger and parking space usage
- Execution Time (s): Computational efficiency

## Complete Workflow

### Phase 1: Hyperparameter Optimization (8-30 hours)
```bash
# Execute complete Scatter Search optimization
python -m src.main --mode scatter_search
```

### Phase 2: Identify Top Performing Archetypes
```bash
# Review optimization results
cat ./results/scatter_search/optimization_report.txt

# Examine top configurations
ls ./results/scatter_search/configurations/
```

### Phase 3: Comprehensive System Evaluation
```bash
# Evaluate top 3 archetypes across all systems
for model in efficiency_focused_rank_1 balanced_optimizer_rank_1 satisfaction_maximizer_rank_1; do
  python -m src.main --mode solution_scatter_search \
    --model_name $model --all_systems
done
```

### Phase 4: MILP Refinement (Optional)
```bash
# Apply hybrid optimization for further improvement
python -m src.main --mode optimize \
  --model_name efficiency_focused_rank_1 \
  --all_systems --alpha_cost 0.7 --alpha_satisfaction 0.3
```

### Phase 5: Final Analysis and Reporting
```bash
# Generate comprehensive analysis report
python -m src.scatter_search.advanced_analysis --base_dir ./results/scatter_search
```

## Advanced Configuration

### DQN Hyperparameters (`hyperparameters.yaml`)
```yaml
learning_rate: 0.0005      # Learning rate for neural network
gamma: 0.99                # Discount factor for future rewards
epsilon_start: 1.0         # Initial exploration rate
epsilon_min: 0.01          # Minimum exploration rate
epsilon_decay: 0.995       # Exploration decay rate
batch_size: 64             # Training batch size
target_update_frequency: 10 # Target network update frequency
memory_size: 100000        # Experience replay buffer size
dueling: true              # Enable Dueling DQN architecture
```

### Scatter Search Parameter Ranges (`hyperparameter_ranges.yaml`)
```yaml
dqn_hyperparameters:
  learning_rate: {type: float, min: 0.00005, max: 0.01}
  gamma: {type: float, min: 0.90, max: 0.9999}
  batch_size: {type: int_discrete, values: [8,16,32,64,128,256]}

reward_weights:
  energy_satisfaction_weight: {type: float, min: 0.2, max: 5.0}
  energy_cost_weight: {type: float, min: 0.01, max: 2.0}
  penalty_skipped_vehicle: {type: float, min: 5.0, max: 500.0}
  reward_assigned_vehicle: {type: float, min: 1.0, max: 200.0}

archetypes:
  cost_minimizer:
    description: "Extreme cost minimization focus"
    energy_cost_weight: [1.0, 2.0]
    energy_satisfaction_weight: [0.2, 0.9]
  satisfaction_maximizer:
    description: "Customer satisfaction above all else"
    energy_satisfaction_weight: [3.6, 5.0]
    energy_cost_weight: [0.01, 0.45]
```

## Troubleshooting Guide

### Common Issues and Solutions

**No checkpoints found error:**
```bash
# Verify directory structure
ls -la ./results/scatter_search/checkpoints/
mkdir -p ./results/scatter_search/checkpoints
```

**CUDA out of memory:**
```bash
# Reduce batch size in configuration
sed -i 's/batch_size: 128/batch_size: 32/g' src/configs/hyperparameters.yaml
```

**Scatter Search performance issues:**
```bash
# Use fast configuration for testing
cp src/configs/scatter_search_config.yaml src/configs/scatter_search_config_fast.yaml
# Edit: population_size: 50, max_time_hours: 2
```

**System configuration not found:**
```bash
# Verify test system data files
ls -la src/configs/system_data/test_system_*.json
```

### Logging and Debugging

```bash
# Monitor Scatter Search progress
tail -f ./results/scatter_search/checkpoints/training_log.txt

# Check optimization convergence
cat ./results/scatter_search/optimization_report.txt

# Review DQN training logs
tail -f ./results/dqn_agent_system_1/logs/dqn_episode_log_system_1.json
```

### Checkpoint Recovery

```bash
# List available checkpoints
ls -la ./results/scatter_search/checkpoints/scatter_checkpoint_iter_*.json

# Resume automatically (prompts user)
python -m src.main --mode scatter_search

# Inspect checkpoint state
python -c "
import json
with open('./results/scatter_search/checkpoints/scatter_checkpoint_iter_XX.json', 'r') as f:
    data = json.load(f)
    print(f'Iteration: {data[\"iteration\"]}')
    print(f'Best solutions: {len(data[\"best_solutions\"])}')
"
```

## Results Interpretation

### Optimization Report Analysis (`optimization_report.txt`)
```
TOP SOLUTIONS FOUND:
1. EFFICIENCY_FOCUSED
   Fitness: 1247.83
   Learning Rate: 0.002341
   Satisfaction Weight: 2.45
   Cost Weight: 0.156

ARCHETYPE ANALYSIS:
COST_MINIMIZER: 23 solutions (avg fitness: 1156.7)
BALANCED_OPTIMIZER: 31 solutions (avg fitness: 1203.4)
```

### System Evaluation Metrics (`config_X.json`)
```json
{
  "energy_performance": {
    "overall_satisfaction_pct": 94.2,
    "total_energy_delivered_kwh": 487.3,
    "energy_deficit_kwh": 12.1
  },
  "economic_performance": {
    "total_energy_cost": 156.78,
    "cost_efficiency": 0.322
  },
  "vehicle_performance": {
    "vehicles_assigned": 38,
    "vehicles_total": 42,
    "assignment_ratio_pct": 90.5
  }
}
```

## Technical Implementation Details

### Deep Q-Network Architecture
The DQN implementation features:
- **Dueling Architecture**: Separate value and advantage streams for improved learning stability
- **Experience Replay**: Large memory buffer (up to 200,000 experiences) for stable training
- **Target Network**: Periodic updates every 10-200 steps for training stability
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Batch Normalization**: Improved convergence and training stability

### Scatter Search Algorithm
The metaheuristic optimization includes:
- **Reference Set Management**: Elite and diverse solution maintenance
- **Cosine Similarity Metrics**: Advanced diversity measurement in hyperparameter space
- **Adaptive Thresholds**: Dynamic similarity thresholds based on convergence progress
- **Multi-level Evaluation**: Fast, medium, and full evaluation stages for efficiency
- **Archetype Classification**: Automatic behavioral pattern recognition

### MILP Formulation
The multi-objective optimization features:
- **Epsilon-Constraint Method**: Convert multi-objective to single-objective with constraints
- **Binary Variables**: Charger assignment and parking slot allocation
- **Continuous Variables**: Power allocation and energy delivery
- **Slack Variables**: Handling infeasible solutions gracefully
- **Dynamic Pricing**: Time-varying electricity costs integration

## Dependencies and Requirements

### Core Dependencies
```
torch==2.7.0              # PyTorch with CUDA 12.6 support
numpy==2.1.2              # Numerical computing
pandas==2.2.3             # Data analysis and manipulation
matplotlib==3.10.3        # Visualization and plotting
seaborn==0.13.2           # Statistical data visualization
PuLP==3.1.1               # Linear programming optimization
scikit-learn==1.6.1       # Machine learning utilities
PyYAML==6.0.2             # YAML configuration file handling
```

But everything that you require is on the requirements.txt, just install it on the root as:

```
pip install -r requirements.txt
```

### GPU Support
The system automatically detects and utilizes NVIDIA GPUs when available:
- CUDA 12.6+ support through PyTorch
- Automatic fallback to CPU computation
- Memory management for large-scale optimization

## Research Background and References

This implementation builds upon recent advances in EV charging optimization and reinforcement learning:

1. **Multi-Agent Reinforcement Learning for EV Scheduling**: Decentralized collaborative approaches for optimal scheduling in charging stations, addressing scalability and coordination challenges.

2. **V2G Integration and Dynamic Pricing**: Optimal scheduling strategies incorporating vehicle-to-grid capabilities with time-varying electricity prices in coupled power-transportation networks.

3. **Robust Scheduling Under Uncertainty**: Advanced methods for handling uncertain demands, deadlines, and system parameters in real-world charging scenarios.

4. **Heuristic Optimization Methods**: Time-dependent personal scheduling problems solved using graph-search algorithms and metaheuristic approaches.

5. **Intelligent Energy Management**: Comprehensive algorithms considering multiple EV charging modes and operational constraints.

6. **Scatter Search for EV Applications**: Metaheuristic optimization specifically adapted for electric vehicle charging scheduling problems.

7. **Deep Reinforcement Learning in Energy Markets**: Multi-DQN approaches for bidding strategies and market participation in electricity spot markets.

8. **Neural Network Integration**: Hybrid approaches combining neural network classifiers with reinforcement learning for efficient autonomous scheduling.

## Performance Benchmarks

### Typical Performance Metrics
- **Optimization Time**: 8-30 hours for complete Scatter Search (200 population, 50 iterations)
- **Memory Usage**: 4-12 GB RAM depending on population size and system complexity
- **GPU Acceleration**: 3-5x speedup with CUDA-enabled hardware
- **Convergence Rate**: 80-95% of final fitness achieved within first 30 iterations
- **Solution Quality**: 15-40% improvement over baseline heuristic methods

### Scalability Characteristics
- **System Size**: Tested on configurations from 5-50 charging points
- **Vehicle Load**: Handles 10-200 vehicles per simulation episode
- **Time Horizon**: Supports 24-168 hour scheduling windows
- **Hyperparameter Space**: Optimizes across 10+ dimensional continuous/discrete spaces

## Future Enhancements

### Planned Improvements
- **Distributed Computing**: Multi-node Scatter Search for larger populations
- **Advanced Neural Architectures**: Transformer-based and attention mechanisms
- **Real-time Adaptation**: Online learning and model updates
- **Multi-objective Visualization**: Pareto frontier analysis tools
- **Uncertainty Quantification**: Bayesian approaches for robust optimization

### Integration Possibilities
- **Smart Grid Integration**: Real-time grid state and pricing data
- **Weather Forecasting**: Solar/wind generation prediction integration
- **Traffic Patterns**: Transportation network optimization coupling
- **Market Mechanisms**: Auction-based and peer-to-peer energy trading

## License and Citation

### License
This project is released under the MIT License. See LICENSE file for full details.

### Citation
If you use this work in your research, please cite:

```bibtex
@software{acosta_bernal_2024_ev_charging,
  author = {Acosta Bernal, Tomás},
  title = {EV Charging Management System with Reinforcement Learning and Multi-Objective Optimization},
  year = {2024},
  url = {https://github.com/username/ev-charging-management},
  note = {Research implementation combining Scatter Search, DQN, and MILP optimization}
}
```

## Contributing

Contributions are welcome through GitHub pull requests. Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass before submission
- Documentation is updated for new features
- Performance benchmarks are provided for algorithmic changes

## Contact

For questions, issues, or collaboration opportunities:
- **Email**: t.acosta@uniandes.edu.co
- **GitHub**: Create an issue for bug reports or feature requests
- **Research Gate**: [Author Profile] for academic discussions

---

**Note**: This system is designed for research and development purposes. Thorough testing and validation are recommended before deployment in production environments.