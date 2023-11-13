import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import argparse

def plot(args, node: int = None):
    output_dir = args.output_dir
    coroutine = args.coroutine
    setting = args.setting
    
    # Load timing information
    execution = 'coroutine' if coroutine else 'sync'
    stats_f = f'{output_dir}/timing_info_{execution}_{setting}_node{node}.json' if node is not None else f'{output_dir}/timing_info_{execution}_{setting}.json'
    with open(stats_f, 'r') as f:
        timing_info = json.load(f)

    # Normalize the times by the smallest timestamp
    min_time = min(min(times) for times in timing_info.values())
    timing_info = {k: [t - min_time for t in v] for k, v in timing_info.items()}

    # Extract the number of GPUs based on the keys
    gpus = list(set(int(key.split('_')[0]) for key in timing_info))  # GPUs are 0-indexed
    init_gpu, last_gpu = min(gpus), max(gpus)
    num_gpus = len(gpus)
    # Create a figure and axis to plot on
    fig, ax = plt.subplots(figsize=(20, num_gpus))

    colors = cycle('bgrcmk')  # Cycle through a list of colors for the plot
    idle_dict = {}
    # Plot the timings for each GPU
    for gpu_id in range(len(gpus)):
        start_times = timing_info.get(f"{gpu_id+init_gpu}_start", [])
        end_times = timing_info.get(f"{gpu_id+init_gpu}_end", [])
        idles = [start - end for start, end in zip(start_times[1:], end_times[:-1])]
        idle_dict[f'{gpu_id}'] = idles
        # idle_dict[f'{gpu_id}_stats'] = {
        #     'mean': np.mean(idles),
        #     'var': np.var(idles),
        #     'median': np.median(idles),
        # }
        print(f'GPU {gpu_id} idle time statistics: mean {np.mean(idles)}, var {np.var(idles)}, median {np.median(idles)}')
        color = next(colors)
        
        # Plot each task for this GPU, increase the linewidth for a wider bar
        for start, end in zip(start_times, end_times):
            ax.hlines(y=gpu_id + 1, xmin=start, xmax=end, colors=color, linewidth=20, label=f'GPU {gpu_id}' if start == start_times[0] else "")

    # Set plot labels and grid
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GPU')
    ax.set_title('Task Inference Profiling')
    ax.set_yticks(range(1, len(gpus) + 1))
    ax.set_yticklabels([f'GPU {i}' for i in range(len(gpus))])
    ax.grid(True)

    # Reverse the y-axis so that GPU 0 is at the top
    ax.invert_yaxis()

    # Create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{stats_f.split(".")[0]}.png')
    # Show the plot
    plt.show()
    
    # Save idle time statistics
    idle_f = f'{output_dir}/idle_time_{execution}_{setting}_node{node}.json' if node is not None else f'{output_dir}/idle_time_{execution}_{setting}.json'
    with open(idle_f, 'w') as f:
        json.dump(idle_dict, f)
        
        

def plot_mix(args, node: int = None):
    output_dir = args.output_dir
    coroutine = args.coroutine
    setting = args.setting
    
    # Load timing information
    execution = 'coroutine' if coroutine else 'sync'
    stats_f = f'{output_dir}/timing_info_{execution}_{setting}_node{node}.json' if node is not None else f'{output_dir}/timing_info_{execution}_{setting}.json'
    with open(stats_f, 'r') as f:
        timing_info = json.load(f)

    # Normalize the times by the smallest timestamp
    # min_time = min(min(times[0] for times in timing_info.values() if times), default=0)
    min_time = None
    for times_list in timing_info.values():
        for times in times_list:
            if min_time is None or times[0] < min_time:
                min_time = times[0]
    min_time = min_time or 0
    timing_info = {k: [[t[0] - min_time, t[1]] for t in v] for k, v in timing_info.items()}

    # Extract the number of GPUs based on the keys
    gpus = list(set(int(key.split('_')[0]) for key in timing_info))  # GPUs are 0-indexed
    init_gpu, last_gpu = min(gpus, default=0), max(gpus, default=0)
    num_gpus = len(gpus)
    # Create a figure and axis to plot on
    fig, ax = plt.subplots(figsize=(20, num_gpus))

    # Define colors for each operation
    colors = {'forward': 'b', 'forward_loss': 'g', 'backward': 'r'}

    # Plot the timings for each GPU
    for gpu_id in range(len(gpus)):
        start_times = timing_info.get(f"{gpu_id+init_gpu}_start", [])
        end_times = timing_info.get(f"{gpu_id+init_gpu}_end", [])

        # Plot each task for this GPU, increase the linewidth for a wider bar
        for (start, start_label), (end, end_label) in zip(start_times, end_times):
            color = colors.get(start_label, 'k')  # default color is black if label not found
            ax.hlines(y=gpu_id + 1, xmin=start, xmax=end, colors=color, linewidth=20, label=start_label)

    # Set plot labels and grid
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GPU')
    ax.set_title('Task Inference Profiling')
    ax.set_yticks(range(1, len(gpus) + 1))
    ax.set_yticklabels([f'GPU {i}' for i in range(len(gpus))])
    ax.grid(True)

    # Reverse the y-axis so that GPU 0 is at the top
    ax.invert_yaxis()

    # Create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{stats_f.split(".")[0]}.png')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='prof')
    parser.add_argument('--coroutine', action='store_true')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], default='random', help='workload setting')
    parser.add_argument('--node', type=int, default=None, help='node id')
    args = parser.parse_args()
    
    if not args.node:
        # plot(args)
        plot_mix(args)
    else:
        for node in range(args.node):
            # plot(args, node)
            plot_mix(args, node)