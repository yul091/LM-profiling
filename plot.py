import json
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import argparse

def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='prof')
    parser.add_argument('--coroutine', action='store_true')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    coroutine = args.coroutine
    
    # Load timing information
    stats_f = f'{output_dir}/timing_info_coroutine.json' if coroutine else f'{output_dir}/timing_info.json'
    with open(stats_f, 'r') as f:
        timing_info = json.load(f)

    # Normalize the times by the smallest timestamp
    min_time = min(min(times) for times in timing_info.values())
    timing_info = {k: [t - min_time for t in v] for k, v in timing_info.items()}

    # Extract the number of GPUs based on the keys
    num_gpus = max(int(key.split('_')[0]) for key in timing_info) + 1  # GPUs are 0-indexed

    # Create a figure and axis to plot on
    fig, ax = plt.subplots(figsize=(20, num_gpus))

    colors = cycle('bgrcmk')  # Cycle through a list of colors for the plot

    # Plot the timings for each GPU
    for gpu_id in range(num_gpus):
        start_times = timing_info.get(f"{gpu_id}_start", [])
        end_times = timing_info.get(f"{gpu_id}_end", [])
        color = next(colors)
        
        # Plot each task for this GPU, increase the linewidth for a wider bar
        for start, end in zip(start_times, end_times):
            ax.hlines(y=gpu_id + 1, xmin=start, xmax=end, colors=color, linewidth=20, label=f'GPU {gpu_id}' if start == start_times[0] else "")

    # Set plot labels and grid
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('GPU')
    ax.set_title('Task Inference Profiling')
    ax.set_yticks(range(1, num_gpus + 1))
    ax.set_yticklabels([f'GPU {i}' for i in range(num_gpus)])
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


if __name__ == '__main__':
    plot()