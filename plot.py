import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import argparse
        
        

def plot_mix(args, ax: plt.Axes, node: int = None, start_time: float = None):
    output_dir = args.output_dir
    setting = args.setting
    priority = args.priority
    workload = args.workload
    retraining_rate = args.retraining_rate
    model_name = args.model_name
    load_balancing = args.load_balancing
    length_distribution = args.length_distribution
    length_heterogeneity = f"_hetero{args.length_heterogeneity}" if args.length_heterogeneity is not None else "_hetero_default"
    active_selection = f"_active{args.active_selection}" if args.active_selection is not None else "_active_1.0"
    if setting == 'isolated':
        isolated_split = args.isolated_split if args.isolated_split is not None else retraining_rate
        setting = f"isolated-split{isolated_split}"
    
    # Load timing information
    if args.test_asyncio:
        stats_f = f'{output_dir}/test_asyncio.json'
    elif node is None:
        stats_f = f'{output_dir}/timing_info_{model_name}_{setting}.json'
    else:
        stats_f = f'{output_dir}/timing_info_{model_name}_{load_balancing}_{setting}-{priority}_{workload}-{length_heterogeneity}-{length_distribution}_{retraining_rate}-{active_selection}_node{node}.json'

    with open(stats_f, 'r') as f:
        timing_info = json.load(f)

    if not timing_info:
        print(f"No timing information found in {stats_f}")
        return
    
    # print(f"Plotting timing information {timing_info}")
    # Normalize the times by the smallest timestamp
    min_time = start_time if start_time is not None else 0
    timing_info = {k: [[t[0] - min_time, t[1]] for t in v] for k, v in timing_info.items()}

    # Extract the number of GPUs based on the keys
    gpus = list(set(int(key.split('_')[0]) for key in timing_info))  # GPUs are 0-indexed
    init_gpu, last_gpu = min(gpus, default=0), max(gpus, default=0)
    num_gpus = len(gpus)

    # Define colors for each operation
    # base_colors = {'forward': 'b', 'forward_grad': 'g', 'backward': 'r'}
    base_colors = {
        'forward': np.array([0, 0, 1]),  # Blue
        'forward_grad': np.array([0, 1, 0]),  # Green
        'backward': np.array([1, 0, 0])  # Red
    }
    lighten_factor = 0.5
    idle_dict = {}
    latencies = defaultdict(list)
    # Plot the timings for each GPU
    for gpu_id in range(num_gpus):
        min_t, max_t = float('inf'), float('-inf')
        start_times = timing_info.get(f"{gpu_id+init_gpu}_start", [])
        end_times = timing_info.get(f"{gpu_id+init_gpu}_end", [])
        if len(start_times) == 1:
            idles = [0]
        else:
            idles = [start - end for (start, start_label), (end, end_label) in zip(start_times[1:], end_times[:-1])]
        idle_dict[f'{gpu_id}'] = idles
        # print(f'GPU {gpu_id} idle time statistics: mean {np.mean(idles)}, var {np.var(idles)}, median {np.median(idles)}')
        # Get the tasks for this GPU
        tasks = list(zip(start_times, end_times))
        num_tasks = len(tasks)
        # Plot each task for this GPU, increase the linewidth for a wider bar
        for i, ((start, start_label), (end, end_label)) in enumerate(tasks):
            base_color = base_colors.get(start_label, np.array([0, 0, 0]))  # default color is black if label not found
            # This will darken the color for each subsequent task
            color = base_color + (np.array([1, 1, 1]) - base_color) * (i / num_tasks) * lighten_factor
            color = np.clip(color, 0, 1)
            ax.hlines(y=gpu_id + 1, xmin=start, xmax=end, linewidth=20, colors=color, label=start_label)
            latencies[start_label].append(end - start)
            min_t = min(min_t, start)
            max_t = max(max_t, end)
        for start_label in latencies.keys():
            if start_label == 'backward' and gpu_id != num_gpus - 1:
                continue

    # Set plot labels and grid
    ax.set_xlabel('Time (s)')
    if node is None:
        ax.set_ylabel('GPU', fontsize=7*num_gpus)
    else:
        ax.set_ylabel(f'Node {node+1}', fontsize=7*num_gpus)
    # ax.set_title('Task Inference Profiling')
    ax.set_yticks(range(1, num_gpus + 1))
    ax.set_yticklabels([f'P{i}' for i in range(num_gpus)], fontsize=5*num_gpus)
    ax.set_ylim(0.5, num_gpus + 0.5)
    ax.grid(True)

    # Reverse the y-axis so that GPU 0 is at the top
    ax.invert_yaxis()

    # Create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='prof')
    parser.add_argument('--model_name', type=str, default='dialogpt', help='model name')
    parser.add_argument('--setting', type=str, default='active', help='workload setting')
    parser.add_argument('--isolated_split', type=float, default=None, help='split ratio for isolated test and train nodes')
    parser.add_argument('--priority', type=str, default='FIFO', help='scheduling priority')
    parser.add_argument('--load_balancing', type=str, default='random', choices=['random', 'workload'], help='node level scheduling policy')
    parser.add_argument('--workload', type=str, choices=['poisson', 'all'], default='poisson', help='workload type')
    parser.add_argument('--length_distribution', type=str, default='random', choices=['random', 'ascending', 'descending', 'bursty'], help='distribution of input sequence length')
    parser.add_argument('--node', type=int, default=1, help='number of nodes for distributed systems')
    parser.add_argument('--retraining_rate', type=float, default=0.1, help='retraining rate')
    parser.add_argument('--length_heterogeneity', type=int, default=None, help='standard deviation of the length distribution of the sampled subset')
    parser.add_argument('--active_selection', type=float, default=None, help='active selection ratio for training tasks')
    parser.add_argument('--test_asyncio', action='store_true')
    args = parser.parse_args()
    
    start_time = None
    output_dir = args.output_dir
    setting = args.setting
    priority = args.priority
    workload = args.workload
    model_name = args.model_name
    retraining_rate = args.retraining_rate
    load_balancing = args.load_balancing
    length_distribution = args.length_distribution
    length_heterogeneity = f"_hetero{args.length_heterogeneity}" if args.length_heterogeneity is not None else "_hetero_default"
    active_selection = f"_active{args.active_selection}" if args.active_selection is not None else "_active_1.0"
    if setting == 'isolated':
        isolated_split = args.isolated_split if args.isolated_split is not None else retraining_rate
        setting = f"isolated-split{isolated_split}"
    
    for node in range(args.node):
        # Load timing information
        stats_f = f'{output_dir}/timing_info_{model_name}_{load_balancing}_{setting}-{priority}_{workload}-{length_heterogeneity}-{length_distribution}_{retraining_rate}-{active_selection}_node{node}.json'

        with open(stats_f, 'r') as f:
            timing_info = json.load(f)
        if not timing_info:
            continue
        gpus = list(set(int(key.split('_')[0]) for key in timing_info))  # GPUs are 0-indexed
        for times_list in timing_info.values():
            for times in times_list:
                if start_time is None or times[0] < start_time:
                    start_time = times[0]
                    
    # fig.subplots_adjust(hspace=0)  # Adjust this value as needed to reduce the gap
    fig, axes = plt.subplots(args.node, 1, figsize=(20, args.node * len(gpus)/1.3), sharex=True)
    for node in range(args.node):
        # plot(args, node)
        ax = axes[node] if args.node > 1 else axes
        plot_mix(args, ax, node, start_time)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_{load_balancing}_{setting}-{priority}_{workload}-{length_heterogeneity}-{length_distribution}_{retraining_rate}-{active_selection}.png", bbox_inches='tight', dpi=300)