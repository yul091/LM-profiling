import os
import json
from collections import defaultdict
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
        
        

def plot_mix(args, ax: plt.axis, node: int = None, start_time: float = None):
    output_dir = args.output_dir
    coroutine = args.coroutine
    setting = args.setting
    workload = args.workload
    retraining_rate = args.retraining_rate
    
    # Load timing information
    execution = 'coroutine' if coroutine else 'sync'
    stats_f = f'{output_dir}/timing_info_{execution}_{setting}_{workload}_{retraining_rate}_node{node}.json' if node is not None else f'{output_dir}/timing_info_{execution}_{setting}.json'
    with open(stats_f, 'r') as f:
        timing_info = json.load(f)

    # Normalize the times by the smallest timestamp
    min_time = start_time if start_time is not None else 0
    timing_info = {k: [[t[0] - min_time, t[1]] for t in v] for k, v in timing_info.items()}

    # Extract the number of GPUs based on the keys
    gpus = list(set(int(key.split('_')[0]) for key in timing_info))  # GPUs are 0-indexed
    init_gpu, last_gpu = min(gpus, default=0), max(gpus, default=0)
    num_gpus = len(gpus)

    # Define colors for each operation
    colors = {'forward': 'b', 'forward_loss': 'g', 'backward': 'r'}
    idle_dict = {}
    latencies = defaultdict(list)
    # Plot the timings for each GPU
    for gpu_id in range(len(gpus)):
        min_t, max_t = float('inf'), float('-inf')
        start_times = timing_info.get(f"{gpu_id+init_gpu}_start", [])
        end_times = timing_info.get(f"{gpu_id+init_gpu}_end", [])
        idles = [start - end for (start, start_label), (end, end_label) in zip(start_times[1:], end_times[:-1])]
        idle_dict[f'{gpu_id}'] = idles
        print(f'GPU {gpu_id} idle time statistics: mean {np.mean(idles)}, var {np.var(idles)}, median {np.median(idles)}')
        # Plot each task for this GPU, increase the linewidth for a wider bar
        for (start, start_label), (end, end_label) in zip(start_times, end_times):
            color = colors.get(start_label, 'k')  # default color is black if label not found
            ax.hlines(y=gpu_id + 1, xmin=start, xmax=end, colors=color, linewidth=20, label=start_label)
            latencies[start_label].append(end - start)
            min_t = min(min_t, start)
            max_t = max(max_t, end)
        for start_label in latencies.keys():
            if start_label == 'backward' and gpu_id != num_gpus - 1:
                continue
            print(f'\t{start_label} latency statistics: mean {np.mean(latencies[start_label])}, var {np.var(latencies[start_label])}, median {np.median(latencies[start_label])}')
        
        total_idles = sum(idles)
        print(f'\tTotal latency: {max_t - min_t}')
        print(f'\tTotal idle time: {total_idles}')
        print(f'\tRubble rate: {total_idles / (max_t - min_t)}')
    

    # Set plot labels and grid
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Node {node+1}', fontsize=20)
    # ax.set_title('Task Inference Profiling')
    ax.set_yticks(range(1, num_gpus + 1))
    ax.set_yticklabels([f'P{i}' for i in range(num_gpus)], fontsize=15)
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
    parser.add_argument('--coroutine', action='store_true')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], default='random', help='workload setting')
    parser.add_argument('--workload', type=str, choices=['poisson', 'all'], default='poisson', help='workload type')
    parser.add_argument('--node', type=int, default=None, help='number of nodes for distributed systems')
    parser.add_argument('--retraining_rate', type=float, default=0.1, help='retraining rate')
    args = parser.parse_args()
    
    if not args.node:
        # plot(args)
        plot_mix(args)
    else:
        start_time = None
        output_dir = args.output_dir
        coroutine = args.coroutine
        setting = args.setting
        workload = args.workload
        retraining_rate = args.retraining_rate
            
        for node in range(args.node):
            # Load timing information
            execution = 'coroutine' if coroutine else 'sync'
            stats_f = f'{output_dir}/timing_info_{execution}_{setting}_{workload}_{retraining_rate}_node{node}.json' if node is not None else f'{output_dir}/timing_info_{execution}_{setting}.json'
            with open(stats_f, 'r') as f:
                timing_info = json.load(f)
                
            for times_list in timing_info.values():
                for times in times_list:
                    if start_time is None or times[0] < start_time:
                        start_time = times[0]
                        
        gpus = list(set(int(key.split('_')[0]) for key in timing_info))  # GPUs are 0-indexed
        fig, axes = plt.subplots(args.node, 1, figsize=(20, args.node * len(gpus)/1.3), sharex=True)
        # fig.subplots_adjust(hspace=0)  # Adjust this value as needed to reduce the gap
            
        for node in range(args.node):
            # plot(args, node)
            ax = axes[node] if args.node > 1 else axes
            plot_mix(args, ax, node, start_time)
            
        # Show the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_info_{execution}_{setting}_{workload}_{retraining_rate}.png")
        plt.show()