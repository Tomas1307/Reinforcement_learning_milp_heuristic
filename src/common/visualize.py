import json
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from collections import defaultdict


def plot_dqn_learning_progress(episode_data_log: list, save_dir: str, title: str = "DQN Learning Progress"):
    """
    Generates and saves plots of training progress for an individual DQN agent.
    Shows total reward, energy satisfaction percentage, and epsilon evolution.

    Args:
        episode_data_log (list): A list of dictionaries, where each dictionary contains
                                 the metrics for an episode (e.g., the result of SimpleEpisodeLogger.get_episode_data()).
        save_dir (str): The directory where the plots will be saved.
        title (str, optional): The main title for the plots. Defaults to "DQN Learning Progress".
    """
    if not episode_data_log:
        print(f"No episode data to plot progress for {title}.")
        return

    episodes = [d['episode'] for d in episode_data_log]
    rewards = [d['reward'] for d in episode_data_log]
    satisfactions = [d['satisfaction_pct'] for d in episode_data_log]
    epsilons = [d['epsilon'] for d in episode_data_log]
    assign_ratios = [d['assign_ratio'] for d in episode_data_log]

    fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(title, fontsize=18)

    axs[0].plot(episodes, rewards, label='Total Reward', color='blue')
    axs[0].set_ylabel('Reward', fontsize=12)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].set_title('Total Reward per Episode', fontsize=14)

    axs[1].plot(episodes, satisfactions, label='Energy Satisfaction (%)', color='green')
    axs[1].set_ylabel('Satisfaction (%)', fontsize=12)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_title('Energy Satisfaction per Episode', fontsize=14)

    axs[2].plot(episodes, assign_ratios, label='Assignment Ratio (%)', color='purple')
    axs[2].set_ylabel('Assignment Ratio (%)', fontsize=12)
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].set_title('Vehicle Assignment Ratio per Episode', fontsize=14)

    axs[3].plot(episodes, epsilons, label='Epsilon (Exploration Rate)', color='red', linestyle='--')
    axs[3].set_xlabel('Episode', fontsize=12)
    axs[3].set_ylabel('Epsilon', fontsize=12)
    axs[3].legend()
    axs[3].grid(True, linestyle='--', alpha=0.6)
    axs[3].set_title('Epsilon Decay over Episodes', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    filename = os.path.join(save_dir, "dqn_learning_progress.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"DQN progress plot saved to: {filename}")


def visualize_solution(schedule: list, config: dict, save_dir: str = None, filename: str = "schedule_plot.png"):
    """
    Visualizes the electric vehicle charging schedule.
    Generates a Gantt chart to display vehicle-to-charger assignments.

    Args:
        schedule (list): A list of tuples/lists representing the schedule:
                         [(ev_id, t_idx, charger_id, slot_idx, power), ...]
        config (dict): The system configuration including times, chargers, arrivals.
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        filename (str, optional): Filename for the plot if saved.
    """
    if not schedule:
        print("No schedule to visualize.")
        return

    times = config["times"]
    
    if "chargers" in config:
        chargers_info = config["chargers"]
    elif "parking_config" in config and "chargers" in config["parking_config"]:
        chargers_info = config["parking_config"]["chargers"]
    else:
        print("Error: No charger information found in the configuration.")
        return
    
    arrivals_info = config["arrivals"]
    
    charger_ids = []
    for c in chargers_info:
        if "charger_id" in c:
            charger_ids.append(c["charger_id"])
        elif "id" in c:
            charger_ids.append(c["id"])
        else:
            print(f"Warning: No charger ID found in: {c}")
    
    charger_ids = sorted(charger_ids)
    ev_ids = sorted(list(set([s[0] for s in schedule])))

    cmap = plt.cm.get_cmap('tab20', len(ev_ids))
    ev_colors = {ev_id: cmap(i) for i, ev_id in enumerate(ev_ids)}

    charger_schedules = defaultdict(list)
    for ev_id, t_idx, charger_id, slot_idx, power in schedule:
        if t_idx < len(times):
            start_time = times[t_idx]
            duration = config.get("dt", 0.25)
            charger_schedules[charger_id].append({
                'ev_id': ev_id,
                'start': start_time,
                'end': start_time + duration,
                'power': power,
                'color': ev_colors[ev_id]
            })

    fig, ax = plt.subplots(figsize=(15, 8))

    y_pos = np.arange(len(charger_ids))
    for i, charger_id in enumerate(charger_ids):
        for task in charger_schedules[charger_id]:
            ax.barh(y_pos[i], task['end'] - task['start'], left=task['start'],
                    color=task['color'], edgecolor='black', alpha=0.8)
            ax.text(task['start'] + (task['end'] - task['start']) / 2, y_pos[i], 
                    f"EV {task['ev_id']}\n{task['power']:.1f}kW",
                    ha='center', va='center', color='white', fontsize=8, weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Charger {cid}" for cid in charger_ids], fontsize=10)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Charger", fontsize=12)
    ax.set_title("EV Charging Schedule (Gantt Chart)", fontsize=16)
    ax.set_xlim(min(times), max(times) + config.get("dt", 0.25))

    handles = []
    labels = []
    for ev_id, color in ev_colors.items():
        patch = plt.Line2D([0], [0], marker='s', color='w', label=f'EV {ev_id}',
                           markerfacecolor=color, markersize=10)
        handles.append(patch)
        labels.append(f'EV {ev_id}')
    
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Schedule plot saved to: {filepath}")
    
    plt.close(fig)


def load_and_plot_progress(progress_dir="./learning_progress"):
    """
    Loads and visualizes the learning progress of all systems (or Scatter Search runs).
    This function assumes a specific file format that you might need to adapt
    for Scatter Search logs.
    """
    files = glob(os.path.join(progress_dir, "system_*_progress.json"))
    
    if not files:
        print(f"No progress files found in {progress_dir}. (This is for system logs, not individual DQN logs)")
        return
    
    systems_data = {}
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            system_id = data["system_info"]["system_id"]
            systems_data[system_id] = data
    
    print(f"Loaded data from {len(systems_data)} systems")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Agent Learning Progress', fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems_data)))
    
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    for i, (system_id, data) in enumerate(sorted(systems_data.items())):
        episodes = [ep['episode'] for ep in data['episodes_data']]
        satisfactions = [ep['satisfaction_pct'] for ep in data['episodes_data']]
        rewards = [ep['reward'] for ep in data['episodes_data']]
        assign_ratios = [ep['assign_ratio'] for ep in data['episodes_data']]
        
        ax1.plot(episodes, satisfactions, label=f'System {system_id}', color=colors[i])
        ax2.plot(episodes, rewards, label=f'System {system_id}', color=colors[i])
        ax3.plot(episodes, assign_ratios, label=f'System {system_id}', color=colors[i])

    ax1.set_title('Energy Satisfaction')
    ax1.set_ylabel('Satisfaction (%)')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Total Reward')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('Assignment Ratio')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Ratio (%)')
    ax3.legend()
    ax3.grid(True)
    
    output_file = os.path.join(progress_dir, "learning_progress_all_systems.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Overall system learning progress plot saved to: {output_file}")


def visualize_rl_solution(rl_schedule: list, system_config: dict, output_dir: str = None, filename: str = "rl_solution_schedule.png"):
    """
    Visualizes a charging solution generated by an RL agent.
    This is a wrapper for visualize_solution that handles the RL format.

    Args:
        rl_schedule (list): The charging schedule generated by the RL agent.
        system_config (dict): The system configuration.
        output_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        filename (str, optional): Filename for the plot if saved.
    """
    print("Generating RL solution visualization...")
    visualize_solution(rl_schedule, system_config, save_dir=output_dir, filename=filename)
    print(f"RL solution visualization saved to: {os.path.join(output_dir, filename) if output_dir else 'not saved'}")


def visualize_milp_solution(milp_schedule, system_config: dict, output_dir: str = None, filename: str = "milp_solution_schedule.png"):
    """
    Visualizes a charging solution generated by the MILP optimizer.
    Adapts the MILP solution format to be compatible with visualize_solution.

    Args:
        milp_schedule: The charging schedule generated by the MILP optimizer.
                       Can be a list (like RL schedule) or a dictionary.
        system_config (dict): The system configuration.
        output_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        filename (str, optional): Filename for the plot if saved.
    """
    converted_schedule = []
    
    if isinstance(milp_schedule, list):
        print("MILP schedule is already in list format, using directly...")
        converted_schedule = milp_schedule
    elif isinstance(milp_schedule, dict):
        print("Converting MILP schedule from dictionary format...")
        # milp_schedule is a dictionary like: {ev_id: [(t_start, t_end, charger_id, slot, power), ...]}
        for ev_id, assignments in milp_schedule.items():
            for assignment in assignments:
                if len(assignment) == 5:  # (t_start, t_end, charger_id, slot, power)
                    t_start, t_end, charger_id, slot, power = assignment
                    times = system_config["times"]
                    t_idx = 0
                    for idx, t in enumerate(times):
                        if abs(t - t_start) < 1e-5:
                            t_idx = idx
                            break
                    converted_schedule.append((ev_id, t_idx, charger_id, slot, power))
                elif len(assignment) == 3:  # (t_idx, charger_id, power)
                    t_idx, charger_id, power = assignment
                    converted_schedule.append((ev_id, t_idx, charger_id, 0, power))
    else:
        print(f"Error: Unrecognized MILP schedule format: {type(milp_schedule)}")
        return

    print("Generating MILP solution visualization...")
    visualize_solution(converted_schedule, system_config, save_dir=output_dir, filename=filename)
    print(f"MILP solution visualization saved to: {os.path.join(output_dir, filename) if output_dir else 'not saved'}")