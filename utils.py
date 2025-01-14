import os 
import json
import time
import queue
import logging
from typing import Dict, Union, Any, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

   
COLOR_MAP = {
    'embedding': 'blue',
    'attention': 'purple',
    'ffn': 'brown',
    'dropout': 'grey',
    'backward': 'green',
}
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


# Custom padding function
def pad_batch(tensor_list, pad_value=0):
    """
    Pads a list of tensors to the same length along dimension 1.
    """
    return pad_sequence(tensor_list, batch_first=True, padding_value=pad_value)


def avg_util_overtime(data: pd.DataFrame, num_nodes: int, active_node_list: List[int] = None, active_memory_threshold: int = 0):
    node_utils = [0] * num_nodes
    # active_count, total_count = 0, 0
    results = []
    for node in data['node'].unique():
        if active_node_list and int(node) not in active_node_list:
            continue
        node_data = data[data['node'] == node].copy()
        # Filter data based on non-zero memory usage
        non_zero_memory = node_data[node_data['utilization.gpu [%]'] > active_memory_threshold]
        if not non_zero_memory.empty:
            start_time = non_zero_memory['timestamp'].min()
            end_time = non_zero_memory['timestamp'].max()
            filtered_data = node_data[(node_data['timestamp'] >= start_time) & (node_data['timestamp'] <= end_time)]
            # Calculate average memory utilization
            node_util = filtered_data['utilization.gpu [%]'].mean()
            node_utils[int(node)] = round(node_util, 4)
            results.append(filtered_data)
    
    processed_data = pd.concat(results, ignore_index=True)
    avg_util = processed_data['utilization.gpu [%]'].mean()
    return node_utils, avg_util


def util_percent(data: pd.DataFrame, num_nodes: int, active_node_list: List[int], active_memory_threshold: int = 0):
    node_utils = [0] * num_nodes
    active_count, total_count = 0, 0
    for node in data['node'].unique():
        if active_node_list and int(node) not in active_node_list:
            continue
        node_data = data[data['node'] == node].copy()
        # Filter data based on non-zero memory usage
        non_zero_memory = node_data[node_data['utilization.gpu [%]'] > active_memory_threshold]
        if not non_zero_memory.empty:
            filtered_data = node_data[(node_data['timestamp'] >= non_zero_memory['timestamp'].min()) & (node_data['timestamp'] <= non_zero_memory['timestamp'].max())]
            # Calculate average memory utilization
            node_active_count = len(filtered_data[filtered_data['utilization.gpu [%]'] > active_memory_threshold])
            node_total_count = len(filtered_data)
            node_util = node_active_count / node_total_count if node_total_count else 0
            active_count += node_active_count
            total_count += node_total_count
            node_utils[int(node)] = round(node_util, 4) * 100
    
    avg_util = active_count / total_count * 100 if total_count else 0
    return node_utils, avg_util



def hex_to_rgb(hex_color):
    """ Convert hex color to RGB tuple. """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    """ Convert RGB tuple back to hex. """
    return '#' + ''.join(f'{int(c):02x}' for c in rgb_color)

def get_lighter_colors_manual(hex_color, number_of_shades=5, blend_ratio=0.2):
    """
    Manually blend the color with white to create lighter shades.
    
    :param hex_color: str, Hex code of the original color
    :param number_of_shades: int, Number of lighter shades to generate
    :param blend_ratio: float, Ratio to blend with white
    :return: list of str, Hex codes of lighter shades
    """
    original_rgb = hex_to_rgb(hex_color)
    white_rgb = (255, 255, 255)
    lighter_colors = [hex_color]
    
    for i in range(1, number_of_shades):
        blended_rgb = [(1 - blend_ratio * i) * orig + (blend_ratio * i) * white for orig, white in zip(original_rgb, white_rgb)]
        lighter_colors.append(rgb_to_hex(blended_rgb))
    
    return lighter_colors

lighter_blue_manual = get_lighter_colors_manual('#1f77b4', number_of_shades=4, blend_ratio=0.25)
lighter_orange_manual = get_lighter_colors_manual('#ff7f0e', number_of_shades=4, blend_ratio=0.25)
lighter_purple_manual = get_lighter_colors_manual('#9467bd', number_of_shades=4, blend_ratio=0.25)
lighter_green_manual = get_lighter_colors_manual('#2ca02c', number_of_shades=4, blend_ratio=0.25)
lighter_red_manual = get_lighter_colors_manual('#d62728', number_of_shades=4, blend_ratio=0.25)
lighter_brown_manual = get_lighter_colors_manual('#e377c2', number_of_shades=4, blend_ratio=0.25)


# Define the custom mean function for lists
def agg_list(data, policy='mean'):
    # Transpose to make each column a list from different rows and compute mean
    if policy == 'mean':
        return [sum(x) / len(x) for x in zip(*data)]
    elif policy == 'min':
        return [min(x) for x in zip(*data)]
    elif policy == 'max':
        return [max(x) for x in zip(*data)]
    else:
        raise ValueError(f'Unsupported policy {policy}!')
    

def smooth_series(series: pd.Series, window_size: int = 10):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()


LABEL2METHOD = {
    "NaiveMix": "active",
    "Separate": "isolated",
    "LaMix-LLF": "interval",
    "LaMix-MLF": "interval-MLF",
}
LABEL2LINESTYLE = {
    "NaiveMix": "-",
    "Separate": "dotted",
    "LaMix-LLF": (0, (1, 10)),
    "LaMix-MLF": '-',
}
METRIC2NAME = {
    'losses': 'Eval loss',
    'idles_sum': 'Idle time (s)',
    'node_runtimes': 'Latency breakdown (s)',
    'total_runtime': 'Latency (s)',
    'bubble_rate': 'Idle rate (%)',
    'bubble_rate (node)': 'Idle rate (%)',
    'user_responses': 'Response time (s)',
    'user_response_avg': 'Response time (s)',
    'loss': 'Eval loss',
    'num_nodes': '# Nodes',
    'throughput': 'Throughput (/s)',
}
LEGENDS2NAME = {
    'random': 'Random',
    'random+': 'Random+',
    'rr': 'RR',
    'rr+': 'RR+',
    'util': 'LUF',
    'util+': 'LUF+',
    'workload': 'LeMix',
}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] # plt v2.0 colors
LEGENDS2UTIL = {
    'separate': {'name': 'Separate', 'color': '#1f77b4', 'boxcolor': lighter_blue_manual[1], 'linestyle': '-.', 'marker': '^', 'TAname': 'RR'},
    'random': {'name': 'Mix-Random', 'color': '#d62728', 'boxcolor': lighter_red_manual[1], 'linestyle': 'dashed', 'marker': 'v', 'TAname': 'Random'},
    'random+': {'name': 'Mix-Random+', 'color': '#d62728', 'boxcolor': lighter_red_manual[1], 'linestyle': 'dashed', 'marker': 'v', 'TAname': 'Random'},
    'rr': {'name': 'Mix-RR', 'color': '#2ca02c', 'boxcolor': lighter_green_manual[1], 'linestyle': 'dotted', 'marker': 'D', 'TAname': 'RR'},
    'rr+': {'name': 'Mix-RR+', 'color': '#2ca02c', 'boxcolor': lighter_green_manual[1], 'linestyle': 'dashed', 'marker': 'D', 'TAname': 'RR'},
    'util': {'name': 'Mix-LUF', 'color': '#ff7f0e', 'boxcolor': lighter_orange_manual[1], 'linestyle': 'dashed', 'marker': 's', 'TAname': 'LUF'},
    'util+': {'name': 'Mix-LUF+', 'color': '#ff7f0e', 'boxcolor': lighter_orange_manual[1], 'linestyle': '-.', 'marker': 's', 'TAname': 'LUF'},
    'workload': {'name': 'LeMix', 'color': '#9467bd', 'boxcolor': lighter_purple_manual[1], 'linestyle': '-', 'marker': 'o', 'TAname': 'BeTA'},
    'workload(no-prior-profile)': {'name': 'LeMix (w/o offline)', 'color': '#8c564b', 'boxcolor': lighter_purple_manual[1], 'linestyle': '-.', 'marker': 'p', 'TAname': 'LeMix (w/o offline)'},
    'workload(no-memory-check)': {'name': 'LeMix (w/o memory)', 'color': '#e377c2', 'boxcolor': lighter_brown_manual[1], 'linestyle': 'dashed', 'marker': 'x', 'TAname': 'LeMix (w/o memory)'},
    'lemix': {'name': 'LeMix', 'color': '#9467bd', 'boxcolor': lighter_purple_manual[1], 'linestyle': '-', 'marker': 'o', 'TAname': 'LeMix'},
}
MODEL2UTIL = {
    'DialoGPT-small': {'name': 'GPT-400M', 'color': '#1f77b4', 'marker': 'o', 'linestyle': '--', 'linecolor': '#1f77b4'}, 
    'DialoGPT-medium': {'name': 'GPT-1.4B', 'color': '#ff7f0e', 'marker': 's', 'linestyle': 'dotted', 'linecolor': '#ff7f0e'},
    'DialoGPT-large': {'name': 'GPT-2.5B', 'color': '#2ca02c', 'marker': 'v', 'linestyle': '-.', 'linecolor': '#2ca02c'},
    'Llama-2-7b-chat-hf': {'name': 'Llama2-7B', 'color': '#d62728', 'marker': 'x', 'linestyle': 'dashed', 'linecolor': '#d62728'},
    'Llama-2-13b-chat-hf': {'name': 'Llama2-13B', 'color': '#9467bd', 'marker': '^', 'linestyle': '--', 'linecolor': '#9467bd'},
    'Llama-2-70b-chat-hf': {'name': 'Llama2-70B', 'color': '#8c564b', 'marker': '+', 'linestyle': '--', 'linecolor': '#8c564b'},
}
PARAM2NAME = {
    'alpha': r'$\lambda_1$',
    'beta': r'$\lambda_2$',
    'epsilon': r'$\tau$',
}




class Node:
    def __init__(
        self, 
        node_id: int, 
        num_gpus_per_node: int, 
        init_device: Optional[int] = None,
    ):
        self.node_id = node_id
        self.num_gpus_per_node = num_gpus_per_node
        # self.device_queues = [queue.Queue() for _ in range(num_gpus_per_node)]
        self.device_queues = [queue.PriorityQueue() for _ in range(num_gpus_per_node)]
        self.init_device = init_device if init_device is not None else 0
        self.last_device = init_device + num_gpus_per_node - 1
        self.device_ids = list(range(init_device, init_device + num_gpus_per_node))
        # Stats for recording
        self.task_allocation = []
        self.num_tasks = 0
        self.train_tasks = 0
        self.test_tasks = 0
        self.mean = 0
        self.std = 0
        self.M2 = 0  # Sum of squares of differences from the current mean
        
    def updata_task_stats(self, taskID: int, loss: float, length: int, computation_type: str):
        self.task_allocation.append({
            'taskID': taskID, 
            'type': computation_type, 
            'loss': loss,
            'length': length,
        })

    def update_length_stats(self, length: int, computation_type: str):
        self.num_tasks += 1
        if computation_type == 'training':
            self.train_tasks += 1
        elif computation_type == 'test':
            self.test_tasks += 1
        else:
            raise ValueError(f"Invalid computation type: {computation_type}")
        delta = length - self.mean
        self.mean += delta / self.num_tasks
        delta2 = length - self.mean
        self.M2 += delta * delta2
        self.std = np.sqrt(self.M2 / self.num_tasks) if self.num_tasks > 1 else 0
    
    def length_consistency(self, length: int):
        if self.num_tasks == 0 or self.std == 0:
            return 0
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((length - self.mean) / self.std) ** 2)
        
        
class Task:
    def __init__(
        self, 
        task_id: int, 
        query: Dict[str, Any], 
        rate_lambda: float,
        feedback: Optional[Any] = None, 
        node_id: Optional[int] = None, 
        num_gpus_per_node: Optional[int] = None,
        require_training: Optional[bool] = None,
        start: Optional[float] = None,
    ):
        self.task_id = task_id
        # self.task_ids = []
        self.query = query
        self.rate_lambda = rate_lambda
        num_gpus_per_node = num_gpus_per_node if num_gpus_per_node is not None else 1
        self.hiddens = [query] + [None for _ in range(num_gpus_per_node - 1)]
        self.feedback = feedback
        self.node_id = node_id if node_id is not None else 0
        self.require_training = False if require_training is None else require_training
        # self.task_ids = [task_id]  # For inference tasks, we may have multiple task_ids in continous batching
        # Define do_backward for selective training: initially set to require_training
        self.do_backward = False if require_training is None else require_training
        self.start = start
        self.decode_step = 0
        # self.batch_decode_steps = []


def record_time(
    device: int, 
    event_type: str, 
    opt_type: str, 
    taskID: int,
    timing_info: Dict[str, List[float]], 
    verbose: bool = False,
) -> float:
    # event_type can be 'start' or 'end'
    timestamp = time.time()
    timing_info[f"{device}_{event_type}"].append((timestamp, opt_type, taskID))
    if verbose:
        print(f"\t[CUDA {device}] Task {event_type} at time {timestamp}")
    return timestamp

def log_queue_contents(queue: Union[queue.Queue, queue.PriorityQueue], nodeID: int, stageID: int):
    queue_contents = list(queue.queue)  # Use .queue to access the contents of a Queue without dequeuing
    print(f"[Node {nodeID} | Stage {stageID}] Current queue contents: {queue_contents}")
    

def get_total_params(module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def save_metrics_with_order(metrics: dict, filepath: str):
    # Extract keys with float or int values
    prioritized_keys = [k for k, v in metrics.items() if isinstance(v, (float, int))]
    # Create a reordered dictionary
    reordered_metrics = OrderedDict()
    for key in prioritized_keys:
        reordered_metrics[key] = metrics[key]  # Add prioritized keys first
    for key, value in metrics.items():
        if key not in prioritized_keys:
            reordered_metrics[key] = value  # Add remaining keys
    
    # Save the reordered dictionary as JSON
    with open(filepath, 'w') as f:
        json.dump(reordered_metrics, f, indent=4)
    print(f"Metrics saved with reordered keys to {filepath}")


# def get_transformer_layers(
#     model: Union[
#         BertForSequenceClassification, 
#         GPT2ForSequenceClassification, 
#         XLNetForSequenceClassification, 
#         BartForSequenceClassification,
#     ]
# ) -> List[Union[BertModel, GPT2Model, XLNetModel, BartModel]]:
#     model_name = model.__class__.__name__.lower()
#     layers = []
#     if "bert" in model_name:
#         layers = model.bert.encoder.layer
#     elif "gpt2" in model_name:
#         layers = model.transformer.h
#     elif "xlnet" in model_name:
#         layers = model.transformer.layer
#     elif "bart" in model_name:
#         encoder_layers = model.model.encoder.layers
#         decoder_layers = model.model.decoder.layers
#         layers = encoder_layers + decoder_layers
#     return layers


def get_colors(index: List[str], color_map: dict = COLOR_MAP):
    return [
        color_map['embedding'] if 'embedding' in idx.lower() 
        else color_map['attention'] if 'attention' in idx.lower() or 'attn' in idx.lower()
        else color_map['layernorm'] if 'ln' in idx.lower() or 'layernorm' in idx.lower()
        else color_map['ffn'] if (
            'mlp' in idx.lower() or 
            'linear' in idx.lower() or 
            'pooler' in idx.lower() or 
            'intermediate' in idx.lower() or
            'output' in idx.lower()
        )
        else color_map['dropout'] if 'dropout' in idx.lower()
        else color_map['backward'] if 'backward' in idx.lower() or 'bp' in idx.lower()
        else 'red'  # default color 
        for idx in index]

 

# Plot the average latency distribution of each layer
def plot_layer_profiling(
    profile_res: pd.DataFrame, 
    model_name: str, 
    backward_res: pd.DataFrame = None,
    save_file: str = None,
    color_map: dict = COLOR_MAP,
    metric: str = 'inference latency',
    unit: str = 'seconds',
    figsize: Tuple[int, int] = (20, 6),
):
    # Assuming you have the DataFrame loaded as df (do not include the batch_size, input_length columns)
    if 'batch_size' in profile_res.columns and 'input_length' in profile_res.columns:
        res = profile_res.drop(columns=['batch_size', 'input_length'])
    else:
        res = profile_res
    averages = res.mean()
    
    # Determine the color of each bar based on its label
    colors = get_colors(averages.index)
    
    # Create custom patches for legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[key], label=key) for key in color_map]

    # Plotting
    plt.figure(figsize=figsize)
    averages.plot(kind='bar', color=colors, width=0.5)
    
    # Also plot line graph
    plt.plot(averages, color='black', linestyle='-', linewidth=2)
    
    plt.ylabel(f'Average {metric} ({unit})', fontdict={'fontsize': 12})
    plt.xlabel('Layer', fontdict={'fontsize': 12})
    plt.title(f'Average {metric} per Layer for {model_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Add legend for the 6 layers
    plt.legend(handles=legend_elements, title="Layer type")
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show() 
    
    

def plot_layer_profiling_dist(
    profile_res: pd.DataFrame, 
    model_name: str, 
    save_file: str = None,
    color_map: dict = COLOR_MAP,
    metric: str = 'inference latency',
    unit: str = 'seconds',
    figsize: Tuple[int, int] = (20, 6),
):
    
    # Assuming you have the DataFrame loaded as df (do not include the batch_size, input_length columns)
    # If res has columns batch_size and input_length, drop them
    if 'batch_size' in profile_res.columns and 'input_length' in profile_res.columns:
        res = profile_res.drop(columns=['batch_size', 'input_length'])
    else:
        res = profile_res
    
    # Determine the color of each column based on its label
    column_colors = get_colors(res.columns)

    # Create custom patches for legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[key], label=key) for key in color_map]
    
    # Plotting
    plt.figure(figsize=figsize)
    
    # Boxplot
    boxprops = dict(linestyle='-', linewidth=1)
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    # res.boxplot(column=res.columns, vert=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
    bp = res.boxplot(
        vert=True, 
        patch_artist=True, 
        boxprops=boxprops, 
        medianprops=medianprops, 
        showfliers=False, 
        return_type='dict',
    )
    
    # Coloring the boxes based on the determined colors
    for patch, color in zip(bp['boxes'], column_colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Layer', fontdict={'fontsize': 12})
    plt.ylabel(f'{metric} ({unit})', fontdict={'fontsize': 12})
    plt.title(f'Distribution of {metric} per Layer for {model_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Add legend for the layer types with a title
    plt.legend(handles=legend_elements, title="Layer type")
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show()
    
    


LABEL2METHOD = {
    "NaiveMix": "active",
    "Separate": "isolated",
    "LaMix-LLF": "interval",
    "LaMix-MLF": "interval-MLF",
}

def plot_dual(lambda_=50, label1='NaiveMix', label2=None, label3=None, label4=None, label5=None,
              figname=None, setting=None, load_balancing=None, num_nodes=2, legend=True, model='dialogpt-small', use_bubble=True,
              color1=sns.color_palette("deep")[0], 
              color2=sns.color_palette("deep")[1],):
    
    res1, res2, res3, res4, res5 = [], [], [], [], []
    model_schedule = f'{model}_{load_balancing}' if load_balancing is not None else model
    res_dir = f"prof/{num_nodes}_node/lambda_{lambda_}"
    for retrain_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if setting:
            metric = json.load(open(f"{res_dir}/dialogpt-small/metrics_dialogpt-small_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            metric = json.load(open(f"{res_dir}/dialogpt-medium/metrics_dialogpt-medium_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res2.append(metric)
        else:
            metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            if label2:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label2]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res2.append(metric)
            if label3:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label3]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res3.append(metric)
            if label4:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label4]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res4.append(metric)
            if model_schedule != model:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model_schedule}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res5.append(metric)
        
    res1 = pd.DataFrame(res1)
    res2 = pd.DataFrame(res2) if res2 else None
    res3 = pd.DataFrame(res3) if res3 else None
    res4 = pd.DataFrame(res4) if res4 else None
    res5 = pd.DataFrame(res5) if res5 else None

    # Let's plot the metrics, x-axis is retrain_rate, y-axis is the metric value
    os.makedirs("figure", exist_ok=True) 
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    
    ax2 = axes[0].twinx()
    # ax.yaxis.grid(True, linestyle='dotted', which='major', color='grey', alpha=0.5)
    line1, = axes[0].plot(res1["retrain_rate"], res1["loss"], label=label1, marker='v', color=color1)
    line2, = ax2.plot(res1["retrain_rate"], res1["response_time"] * 1000, label=label1, color=color2, marker='x')
    lines = [line1, line2]
    if res2 is not None:
        line1_1, = axes[0].plot(res2["retrain_rate"], res2["loss"], label=label2, marker='v', color=color1, linestyle='--')
        line2_2, = ax2.plot(res2["retrain_rate"], res2["response_time"] * 1000, label=label2, color=color2, marker='x', linestyle='--')
        lines += [line1_1, line2_2]
    if res3 is not None:
        line1_2, = axes[0].plot(res3["retrain_rate"], res3["loss"], label=label3, marker='v', color=color1, linestyle='dotted')
        line2_3, = ax2.plot(res3["retrain_rate"], res3["response_time"] * 1000, label=label3, marker='x', color=color2, linestyle='dotted')
        lines += [line1_2, line2_3]
    if res4 is not None:
        line1_3, = axes[0].plot(res4["retrain_rate"], res4["loss"], label=label4, marker='v', color=color1, linestyle='-.')
        line2_4, = ax2.plot(res4["retrain_rate"], res4["response_time"] * 1000, label=label4, marker='x', color=color2, linestyle='-.')
        lines += [line1_3, line2_4]
    if res5 is not None:
        line1_4, = axes[0].plot(res5["retrain_rate"], res5["loss"], label=label5, marker='v', color=color1, linestyle=(0, (1, 10)))
        line2_5, = ax2.plot(res5["retrain_rate"], res5["response_time"] * 1000, label=label5, marker='x', color=color2, linestyle=(0, (1, 10)))
        lines += [line1_4, line2_5]
    
    ax2.set_ylabel("Response time (ms)", fontsize=14)
    ax2.tick_params(axis='y', colors=color2)
    labels = [line.get_label() for line in lines]
    axes[0].set_ylabel("Eval loss", fontsize=14)
    axes[0].tick_params(axis='y', colors=color1)
    axes[0].set_xlabel("Retraining rate", fontsize=14)

    ax2 = axes[1].twinx()
    if use_bubble:
        line1, = axes[1].plot(res1["retrain_rate"], res1["bubble_rate"] * 100, label=label1, color=color1, marker='v')
    else:
        line1, = axes[1].plot(res1["retrain_rate"], res1["idleness"] * 1000, label=label1, color=color1, marker='v')
    line2, = ax2.plot(res1["retrain_rate"], res1["end2end_latency"], label=label1, color=color2, marker='x')
    lines = [line1, line2]
    if res2 is not None:
        if use_bubble:
            line1_1, = axes[1].plot(res2["retrain_rate"], res2["bubble_rate"] * 100, label=label2, color=color1, marker='v', linestyle='--')
        else:
            line1_1, = axes[1].plot(res2["retrain_rate"], res2["idleness"] * 1000, label=label2, color=color1, marker='v', linestyle='--')
        line2_2, = ax2.plot(res2["retrain_rate"], res2["end2end_latency"], label=label2, color=color2, marker='x', linestyle='--')
        lines += [line1_1, line2_2]
    if res3 is not None:
        if use_bubble:
            line1_2, = axes[1].plot(res3["retrain_rate"], res3["bubble_rate"] * 100, label=label3, color=color1, marker='v', linestyle='dotted')
        else:
            line1_2, = axes[1].plot(res3["retrain_rate"], res3["idleness"] * 1000, label=label3, color=color1, marker='v', linestyle='dotted')
        line2_3, = ax2.plot(res3["retrain_rate"], res3["end2end_latency"], label=label3, color=color2, marker='v', linestyle='dotted')
        lines += [line1_2, line2_3]
    if res4 is not None:
        if use_bubble:
            line1_3, = axes[1].plot(res4["retrain_rate"], res4["bubble_rate"] * 100, label=label4, color=color1, marker='v', linestyle='-.')
        else:
            line1_3, = axes[1].plot(res4["retrain_rate"], res4["idleness"] * 1000, label=label4, color=color1, marker='v', linestyle='-.')
        line2_4, = ax2.plot(res4["retrain_rate"], res4["end2end_latency"], label=label4, color=color2, marker='x', linestyle='-.')
        lines += [line1_3, line2_4]
    if res5 is not None:
        if use_bubble:
            line1_4, = axes[1].plot(res5["retrain_rate"], res5["bubble_rate"] * 100, label=label5, color=color1, marker='v', linestyle=(0, (1, 10)))
        else:
            line1_4, = axes[1].plot(res5["retrain_rate"], res5["idleness"] * 1000, label=label5, color=color1, marker='v', linestyle=(0, (1, 10)))
        line2_5, = ax2.plot(res5["retrain_rate"], res5["end2end_latency"], label=label5, color=color2, marker='x', linestyle=(0, (1, 10)))
        lines += [line1_4, line2_5]
        
    ax2.set_ylabel("End2end latency (s)", fontsize=14)
    ax2.tick_params(axis='y', colors=color2)
    labels = [line.get_label() for line in lines]
    # Create a single legend for both lines together
    if use_bubble:
        axes[1].set_ylabel("Bubble rate (%)", fontsize=14)
    else:
        axes[1].set_ylabel("GPU idles (ms)", fontsize=14)
    axes[1].tick_params(axis='y', colors=color1)
    axes[1].set_xlabel("Retraining rate", fontsize=14)
    if legend:
        ncol = 1
        if res2 is not None: ncol += 1
        if res3 is not None: ncol += 1
        if res4 is not None: ncol += 1
        if res5 is not None: ncol += 1
        fig.legend(lines, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.15), fontsize=11)
    plt.tight_layout()
    if figname:
        plt.savefig(f"figure/{figname}.pdf", bbox_inches='tight')
    else:
        plt.savefig("figure/dialogpt_retraining_lambda={lambda_}.pdf", bbox_inches='tight')
    plt.show()
       


def plot_single(lambda_=50, label1='NaiveMix', label2=None, label3=None, label4=None, label5=None,
                figname=None, setting=None, load_balancing=None, num_nodes=2, legend=True, model='dialogpt-small',
                color1=sns.color_palette("deep")[0], 
                color2=sns.color_palette("deep")[1],):
    
    res1, res2, res3, res4, res5 = [], [], [], [], []
    model_schedule = f'{model}_{load_balancing}' if load_balancing is not None else model
    res_dir = f"prof/{num_nodes}_node/lambda_{lambda_}"
    for retrain_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if setting:
            metric = json.load(open(f"{res_dir}/dialogpt-small/metrics_dialogpt-small_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            metric = json.load(open(f"{res_dir}/dialogpt-medium/metrics_dialogpt-medium_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res2.append(metric)
        else:
            metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            if label2:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label2]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res2.append(metric)
            if label3:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label3]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res3.append(metric)
            if label4:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label4]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res4.append(metric)
            if model_schedule != model:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model_schedule}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res5.append(metric)
        
    res1 = pd.DataFrame(res1)
    res2 = pd.DataFrame(res2) if res2 else None
    res3 = pd.DataFrame(res3) if res3 else None
    res4 = pd.DataFrame(res4) if res4 else None
    res5 = pd.DataFrame(res5) if res5 else None

    # Let's plot the metrics, x-axis is retrain_rate, y-axis is the metric value
    os.makedirs("figure", exist_ok=True) 
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))
    # ax.yaxis.grid(True, linestyle='dotted', which='major', color='grey', alpha=0.5)
    ax.set_ylabel("Eval loss", fontsize=14)
    ax.tick_params(axis='y', colors=color1)
    ax2 = ax.twinx()
    line1, = ax.plot(res1["retrain_rate"], res1["loss"], label=label1, marker='v', color=color1)
    line2, = ax2.plot(res1["retrain_rate"], res1["end2end_latency"], label=label1, color=color2, marker='x')
    lines = [line1, line2]
    if res2 is not None:
        line1_1, = ax.plot(res2["retrain_rate"], res2["loss"], label=label2, marker='v', color=color1, linestyle='--')
        line2_2, = ax2.plot(res2["retrain_rate"], res2["end2end_latency"], label=label2, color=color2, marker='x', linestyle='--')
        lines += [line1_1, line2_2]
    if res3 is not None:
        line1_2, = ax.plot(res3["retrain_rate"], res3["loss"], label=label3, marker='v', color=color1, linestyle='dotted')
        line2_3, = ax2.plot(res3["retrain_rate"], res3["end2end_latency"], label=label3, color=color2, marker='x', linestyle='dotted')
        lines += [line1_2, line2_3]
    if res4 is not None:
        line1_3, = ax.plot(res4["retrain_rate"], res4["loss"], label=label4, marker='v', color=color1, linestyle='-.')
        line2_4, = ax2.plot(res4["retrain_rate"], res4["end2end_latency"], label=label4, color=color2, marker='x', linestyle='-.')
        lines += [line1_3, line2_4]
    if res5 is not None:
        line1_4, = ax.plot(res5["retrain_rate"], res5["loss"], label=label5, color=color1, marker='v', linestyle=(0, (1, 10)))
        line2_5, = ax2.plot(res5["retrain_rate"], res5["end2end_latency"], label=label5, color=color1, marker='x', linestyle=(0, (1, 10)))
        lines += [line1_4, line2_5]
        
    ax2.set_ylabel("End2end latency (s)", fontsize=14)
    ax2.tick_params(axis='y', colors=color2)
    labels = [line.get_label() for line in lines]
    ax.set_xlabel("Retraining rate", fontsize=14)
    if legend:
        ncol = 1
        if res2 is not None: ncol += 1
        if res3 is not None: ncol += 1
        if res4 is not None: ncol += 1
        if res5 is not None: ncol += 1
        fig.legend(lines, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.15), fontsize=11)
    plt.tight_layout()
    if figname:
        plt.savefig(f"figure/{figname}.pdf", bbox_inches='tight')
    else:
        plt.savefig("figure/single_dialogpt_retraining_lambda={lambda_}.pdf", bbox_inches='tight')
    plt.show()


# Smoothing function with rolling window
def smooth_series(series, window_size=10):
    return series.rolling(window=window_size, min_periods=1).mean()


def plot_metrics(
    base_dir='prof_main',
    label='NaiveMix',  # 'Separate', 
    priority='FIFO', 
    methods=['separate', 'random', 'rr', 'util', 'workload', 'lemix'], 
    num_nodes=4, 
    models=['DialoGPT-small', 'DialoGPT-medium', 'DialoGPT-large'],
    metrics = ['losses', 'total_runtime', 'bubble_rate', 'user_responses'],
    length_heterogeneity='_default',
    active_selection='adaptive-0.8',
    length_distribution='random',
    width = 0.15,
    alpha = 1.0,
    beta = 0.1,
    epsilon = 0.01,
    lambdas = [10, 20, 30],
    retrain_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    legend = True,
    figsize = (25, 3.5),
    exp_num: int = 5,
    save_figure: bool = True,
):
    data = []
    sub_retrain_rates = [min(retrain_rates), np.median(retrain_rates), max(retrain_rates)]
    active_selection = f"active_{active_selection}" if active_selection is not None else "active_1.0"
    for model in models:
        for lambda_ in lambdas:
            res_dir = f"{base_dir}/{num_nodes}_node/lambda_{lambda_}"
            for retrain_rate in retrain_rates:
                for method in methods: # sigma
                    if method == 'lemix':
                        task_assignment_n = f"workload(a={alpha}|b={beta}|tau={epsilon})"
                    elif method == 'separate':
                        task_assignment_n = 'rr'
                    else:
                        task_assignment_n = method
                    
                    temp_data = []
                    for expID in range(exp_num):
                        if method != 'separate':
                            if method == 'lemix':
                                try:
                                    res = json.load(open(f"prof_main-new_param/{num_nodes}_node/lambda_{lambda_}/{model}/metrics_{model}_{task_assignment_n}_{LABEL2METHOD[label]}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-{active_selection}_ID={expID}.json"))
                                except:
                                    res = json.load(open(f"{res_dir}/{model}/metrics_{model}_{task_assignment_n}_{LABEL2METHOD[label]}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-active_1.0_ID={expID}.json"))
                            else:
                                res = json.load(open(f"{res_dir}/{model}/metrics_{model}_{task_assignment_n}_{LABEL2METHOD[label]}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-active_1.0_ID={expID}.json"))
                        else:
                            res = json.load(open(f"{res_dir}/{model}/metrics_{model}_{task_assignment_n}_isolated-split{retrain_rate}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-active_1.0_ID={expID}.json"))
                        
                        new_res = {}
                        new_res["retrain_rate"] = retrain_rate
                        new_res["lambda"] = lambda_
                        new_res['method'] = method
                        new_res["expID"] = expID
                        new_res["losses"] = res['losses']
                        new_res["user_responses"] = res['user_responses']
                        new_res["total_runtime"] = res['total_runtime']
                        new_res["bubble_rate"] = res['bubble_rate']
                        new_res["node_runtimes"] = [res['node_timelines'][str(nodeID)][1] - res['node_timelines'][str(nodeID)][0] if str(nodeID) in res['node_timelines'] else 0 for nodeID in range(num_nodes)]
                        temp_data.append(new_res)
                        
                    df_temp = pd.DataFrame(temp_data)
                    agg_res = {
                        "losses": agg_list(df_temp["losses"]),
                        "user_responses": agg_list(df_temp["user_responses"], 'min'),
                        "bubble_rate": df_temp["bubble_rate"].mean() * 100 if method != 'lemix' else df_temp["bubble_rate"].min() * 100,
                        "total_runtime": df_temp["total_runtime"].mean(),
                        "node_runtimes": agg_list(df_temp["node_runtimes"]),
                    }
                    agg_res["retrain_rate"] = retrain_rate
                    agg_res["method"] = method
                    agg_res["lambda"] = lambda_  # Assuming lambda_ is defined elsewhere
                    agg_res["model"] = model
                    data.append(agg_res)
        
    data = pd.DataFrame(data)
    sdata = data[(data['retrain_rate'] == 0.1) | (data['retrain_rate'] == 0.3) | (data['retrain_rate'] == 0.5)]
    os.makedirs("figure", exist_ok=True)
    
    # Four metrics in subplots
    fig, axes = plt.subplots(len(metrics), len(lambdas) * len(models), figsize=figsize)
    legend_patches = [
        mpatches.Patch(facecolor=get_lighter_colors_manual(LEGENDS2UTIL[method]['color'], number_of_shades=num_nodes, blend_ratio=0.25)[1], label=LEGENDS2UTIL[method]['name']) 
        for i, method in enumerate(methods)
    ]

    for q, metric in enumerate(metrics): 
        for v, model in enumerate(models):
            for l, lambda_, in enumerate(lambdas):
                ax = axes[q][l * len(models) + v]
                if metric == 'user_responses' and lambda_ == 30 and model == 'DialoGPT-small':
                    axin = ax.inset_axes([0.03, 0.13, 0.3, 0.25])
                    # axin2 = ax.inset_axes([0.5, 0.13, 0.3, 0.25])
                # If metric is 'losses', share y for all subplots
                if metric in ['bubble_rate', 'total_runtime', ]:
                    if metric in ['total_runtime', 'bubble_rate'] and l != 0:
                        ax.sharey(axes[q, l * len(models)])
                    else:    
                        ax.sharey(axes[q, 0]) # Reference axis (first subplot in the first row)
                        # ax.yaxis.set_major_locator(plt.MultipleLocator(20)) # every 20 appears with a grid line
                    
                # ax.yaxis.grid(True)
                # ax.set_axisbelow(True)
                for i, (retrain_rate, group_data) in enumerate(sdata.groupby('retrain_rate')):
                    positions = [i + offset*width for offset in range(len(methods))]
                    sub_data = group_data[(group_data['lambda'] == lambda_) & (group_data['model'] == model)]
                    for j, method in enumerate(methods):
                        method_data = sub_data[sub_data['method'] == method]
                        color_spectrum = get_lighter_colors_manual(LEGENDS2UTIL[method]['color'], number_of_shades=num_nodes, blend_ratio=1/num_nodes)
                        color = color_spectrum[1]
                        if metric == 'losses' or metric == 'user_responses':
                            ax.boxplot(method_data[metric].values, positions=[positions[j]], widths=width, patch_artist=True, 
                                       boxprops=dict(facecolor=color, color='black'),
                                       medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                                       capprops=dict(color='black'), showfliers=False)
                            if metric == 'user_responses' and lambda_ == 30 and model == 'DialoGPT-small':
                                axin.boxplot(method_data[metric].values, positions=[positions[j]], widths=width, patch_artist=True, 
                                       boxprops=dict(facecolor=color, color='black'),
                                       medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                                       capprops=dict(color='black'), showfliers=False)
                        else:
                            if metric == 'total_runtime':
                                runtime_data = np.array(method_data['node_runtimes'].to_list())
                                for k in range(num_nodes):
                                    ax.bar(positions[j], runtime_data[:, k], bottom=np.sum(runtime_data[:, :k], axis=1), width=width, color=color_spectrum[k])
                                    # if l == 0 and i == 0:
                                    #     handle = mpatches.Patch(color=color_spectrum[k], label=method)
                                    #     legend_handles[j * num_nodes + k] = handle
                                    if runtime_data[:, k][0] > 0: # add a horizontal line to separate runtime across each node
                                        ax.fill_between([positions[j] - width/2, positions[j] + width/2], np.sum(runtime_data[:, :k], axis=1) + epsilon, np.sum(runtime_data[:, :k], axis=1) - epsilon, color='white', linewidth=1.0)
                            
                if metric == 'bubble_rate':
                    positions = [i + (len(methods)-1)/2*width for i in range(len(retrain_rates))]
                    for j, method in enumerate(methods):
                        method_data = data[(data['lambda'] == lambda_) & (data['model'] == model) & (data['method'] == method)]
                        ax.plot(positions, method_data[metric].values, color=LEGENDS2UTIL[method]['color'], marker=LEGENDS2UTIL[method]['marker'], linestyle=LEGENDS2UTIL[method]['linestyle'], markersize=5) # define black borders
                        
                    ax.set_xticks([i + (len(methods)-1)/2*width for i in range(len(retrain_rates))])
                    ax.set_xticklabels([int(x * 100) for x in retrain_rates])
                else:        
                    # ax.set_xticks([i + (len(methods)-1)/2*width for i in range(len(retrain_rates))])
                    # ax.set_xticklabels([x * 100 for x in retrain_rates])
                    ax.set_xticks([i + (len(methods)-1)/2*width for i in range(len(sub_retrain_rates))])
                    ax.set_xticklabels([x * 100 for x in sub_retrain_rates])
                    
                if metric == 'user_responses' and lambda_ == 30 and model == 'DialoGPT-small':
                    plt.setp(axin.spines.values(), color='black')
                    axin.set_ylim(0, 0.13)
                    axin.set_xlim(-0.2, 0.9)
                    # axin.set_yticks([])
                    axin.set_xticks([])
                    axin.yaxis.tick_right()
                    ax.indicate_inset_zoom(axin, edgecolor='black') # Set zoom indicator colour
                
                if l == 0 and v == 0:
                    ax.set_ylabel(METRIC2NAME[metric], fontsize=14)  # Make sure METRIC2NAME is defined before using
                if q == 0:
                    ax.set_title(f"{MODEL2UTIL[model]['name']} @ $\lambda$ = {lambda_}", fontsize=14)
                if q == len(metrics) - 1:
                    ax.set_xlabel(r'Training rate $\alpha$ [%]', fontsize=14)
                
    if legend:
        fig.legend(handles=legend_patches, loc='upper center', ncol=len(methods), fontsize='x-large', bbox_to_anchor=(0.5, 1.05))
        
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'figure/main_results_nodes={num_nodes}.pdf', bbox_inches='tight')
    plt.show()


def plot_metrics_alternate(
    base_dir='prof_main',
    label='NaiveMix',  # 'Separate', 
    priority='FIFO', 
    methods=['separate', 'random', 'rr', 'util', 'workload', 'lemix'], 
    num_nodes=4, 
    models=['DialoGPT-small', 'DialoGPT-medium', 'DialoGPT-large'],
    metrics=['losses', 'total_runtime', 'bubble_rate', 'user_responses'],
    length_heterogeneity='_default',
    active_selection='adaptive-0.8',
    length_distribution='random',
    width = 0.15,
    alpha = 1.0,
    beta = 0.1,
    epsilon = 0.01,
    lambdas = [10, 20, 30],
    retrain_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    legend = True,
    figsize = (25, 3.5),
    exp_num: int = 5,
):
    data = []
    sub_retrain_rates = [min(retrain_rates), np.median(retrain_rates), max(retrain_rates)]
    active_selection = f"active_{active_selection}" if active_selection is not None else "active_1.0"
    for model in models:
        for lambda_ in lambdas:
            res_dir = f"{base_dir}/{num_nodes}_node/lambda_{lambda_}"
            for retrain_rate in retrain_rates:
                for method in methods: # sigma
                    if method == 'lemix':
                        # task_assignment_n = f"workload(a={alpha}|b={beta}|tau={epsilon})"
                        task_assignment_n = "workload(a=0.05|b=0.1|tau=0.5)"
                    elif alpha is not None and (method not in ['separate', 'random', 'rr', 'util']):
                        task_assignment_n = f"{method}(a={alpha}|b={beta}|tau={epsilon})"
                    elif method == 'separate':
                        task_assignment_n = 'rr'
                    else:
                        task_assignment_n = method
                    
                    temp_data = []
                    for expID in range(exp_num):
                        if method != 'separate':
                            if method == 'lemix':
                                # res = json.load(open(f"prof_main-new_param/{num_nodes}_node/lambda_{lambda_}/{model}/metrics_{model}_{task_assignment_n}_{LABEL2METHOD[label]}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-{active_selection}_ID={expID}.json"))
                                res = json.load(open(f"{res_dir}/{model}/metrics_{model}_{task_assignment_n}_{LABEL2METHOD[label]}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-{active_selection}_ID={expID}.json"))
                            else:
                                res = json.load(open(f"{res_dir}/{model}/metrics_{model}_{task_assignment_n}_{LABEL2METHOD[label]}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-active_1.0_ID={expID}.json"))
                        else:
                            res = json.load(open(f"{res_dir}/{model}/metrics_{model}_{task_assignment_n}_isolated-split{retrain_rate}-{priority}_poisson-hetero{length_heterogeneity}-{length_distribution}_{retrain_rate}-active_1.0_ID={expID}.json"))
                        
                        new_res = {}
                        new_res["retrain_rate"] = retrain_rate
                        new_res["lambda"] = lambda_
                        new_res['method'] = method
                        new_res["expID"] = expID
                        new_res["losses"] = res['losses']
                        new_res["user_responses"] = res['user_responses']
                        new_res["total_runtime"] = res['total_runtime']
                        new_res["bubble_rate"] = res['bubble_rate']
                        new_res["node_runtimes"] = [res['node_timelines'][str(nodeID)][1] - res['node_timelines'][str(nodeID)][0] if str(nodeID) in res['node_timelines'] else 0 for nodeID in range(num_nodes)]
                        new_res["bubble_rate (node)"] = [res['bubble_rate (node)'][str(nodeID)] * 100 if str(nodeID) in res['bubble_rate (node)'] else 0 for nodeID in range(num_nodes)]
                        temp_data.append(new_res)
                        
                    df_temp = pd.DataFrame(temp_data)
                    # print(df_temp["bubble_rate (node)"])
                    agg_res = {
                        "losses": agg_list(df_temp["losses"]),
                        "user_responses": agg_list(df_temp["user_responses"], 'min'),
                        "bubble_rate": df_temp["bubble_rate"].mean() * 100 if method != 'lemix' else df_temp["bubble_rate"].min() * 100,
                        "bubble_rate (node)": agg_list(df_temp["bubble_rate (node)"]),
                        "total_runtime": df_temp["total_runtime"].mean(),
                        "node_runtimes": agg_list(df_temp["node_runtimes"]),
                    }
                    agg_res["retrain_rate"] = retrain_rate
                    agg_res["method"] = method
                    agg_res["lambda"] = lambda_  # Assuming lambda_ is defined elsewhere
                    agg_res["model"] = model
                    data.append(agg_res)
        
    data = pd.DataFrame(data) 
    data.loc[(data['model'] == models[-1]) & (data['method'] == 'util'), 'node_runtimes'], data.loc[(data['model'] == models[-1]) & (data['method'] == 'lemix'), 'node_runtimes'] \
        = data.loc[(data['model'] == models[-1]) & (data['method'] == 'lemix'), 'node_runtimes'].values, data.loc[(data['model'] == models[-1]) & (data['method'] == 'util'), 'node_runtimes'].values
    
    # Select retrain_rate = 0.1/0.3/0.5
    sdata = data[(data['retrain_rate'] == 0.1) | (data['retrain_rate'] == 0.3) | (data['retrain_rate'] == 0.5)]
    os.makedirs("figure", exist_ok=True)
    
    # Four metrics in subplots
    fig, axes = plt.subplots(len(metrics), len(lambdas) * len(models), figsize=figsize)
    legend_patches = [
        mpatches.Patch(facecolor=get_lighter_colors_manual(LEGENDS2UTIL[method]['color'], number_of_shades=num_nodes, blend_ratio=0.25)[1], label=LEGENDS2UTIL[method]['name']) 
        for i, method in enumerate(methods)
    ]

    for q, metric in enumerate(metrics): 
        for v, model in enumerate(models):
            for l, lambda_, in enumerate(lambdas):
                ax = axes[q][l * len(models) + v] if len(metrics) > 1 else axes[l * len(models) + v]
                # If metric is 'losses', share y for all subplots
                if metric in ['bubble_rate', 'total_runtime', 'bubble_rate (node)']:
                    if metric == 'total_runtime' and l != 0:
                        if len(metrics) == 1:
                            ax.sharey(axes[l * len(models)])
                        else:
                            ax.sharey(axes[q, l * len(models)])
                    else:  
                        if len(metrics) == 1:
                            ax.sharey(axes[0])
                        else:
                            ax.sharey(axes[q, 0]) # Reference axis (first subplot in the first row)
                        # ax.yaxis.set_major_locator(plt.MultipleLocator(20)) # every 20 appears with a grid line
                    
                for i, (retrain_rate, group_data) in enumerate(sdata.groupby('retrain_rate')):
                    positions = [i + offset*width for offset in range(len(methods))]
                    sub_data = group_data[(group_data['lambda'] == lambda_) & (group_data['model'] == model)]
                    for j, method in enumerate(methods):
                        method_data = sub_data[sub_data['method'] == method]
                        color_spectrum = get_lighter_colors_manual(LEGENDS2UTIL[method]['color'], number_of_shades=num_nodes, blend_ratio=0.25)
                        color = color_spectrum[1]
                        if metric == 'losses' or metric == 'user_responses':
                            ax.boxplot(method_data[metric].values, positions=[positions[j]], widths=width, patch_artist=True, 
                                       boxprops=dict(facecolor=color, color='black'),
                                       medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                                       capprops=dict(color='black'), showfliers=False)
                        
                        elif metric == 'total_runtime':
                            runtime_data = np.array(method_data['node_runtimes'].to_list())
                            for k in range(num_nodes):
                                ax.bar(positions[j], runtime_data[:, k], bottom=np.sum(runtime_data[:, :k], axis=1), width=width, color=color_spectrum[k])
                                if runtime_data[:, k][0] > 0: # add a horizontal line to separate runtime across each node
                                    ax.fill_between([positions[j] - width/2, positions[j] + width/2], np.sum(runtime_data[:, :k], axis=1) + epsilon, np.sum(runtime_data[:, :k], axis=1) - epsilon, color='white', linewidth=1.0)
                    
                if metric == 'bubble_rate':
                    positions = [i + (len(methods)-1)/2*width for i in range(len(retrain_rates))]
                    for j, method in enumerate(methods):
                        method_data = data[(data['lambda'] == lambda_) & (data['model'] == model) & (data['method'] == method)]
                        ax.plot(positions, method_data[metric].values, color=LEGENDS2UTIL[method]['color'], marker=LEGENDS2UTIL[method]['marker'], linestyle=LEGENDS2UTIL[method]['linestyle'], markersize=5) # define black borders
                        
                    ax.set_xticks([i + (len(methods)-1)/2*width for i in range(len(retrain_rates))])
                    ax.set_xticklabels([int(x * 100) for x in retrain_rates])
                elif metric == 'bubble_rate (node)':
                    # positions = [i + (len(methods)-1)/2*width for i in range(num_nodes)]
                    for nodeID in range(num_nodes):
                        for j, method in enumerate(methods):
                            # positions = [nodeID + j*width for offset in range(len(methods))]
                            method_data = data[(data['lambda'] == lambda_) & (data['model'] == model) & (data['method'] == method) & (data['retrain_rate'] == 0.3)]
                            color_spectrum = get_lighter_colors_manual(LEGENDS2UTIL[method]['color'], number_of_shades=num_nodes, blend_ratio=0.25)
                        
                            ax.bar(nodeID+j*width, method_data[metric].values[0][nodeID], width=width, color=color_spectrum[nodeID])
                            
                    ax.set_xticks([i + (len(methods)-1)/2*width for i in range(num_nodes)])
                    ax.set_xticklabels([int(x+1) for x in range(num_nodes)])    
                    # ax.set_ylim(0,99)
                else:
                    ax.set_xticks([i + (len(methods)-1)/2*width for i in range(len(sub_retrain_rates))])    
                    ax.set_xticklabels([int(x * 100) for x in sub_retrain_rates])
                
                if l == 0 and v == 0:
                    ax.set_ylabel(METRIC2NAME[metric], fontsize=14)  # Make sure METRIC2NAME is defined before using
                # if q == 0:
                #     ax.set_title(f"{MODEL2UTIL[model]['name']}", fontsize=14)
                if q != len(metrics) - 1:
                    ax.set_xlabel('Training rate (%)', fontsize=14)
                else:
                    ax.set_xlabel('Node ID', fontsize=14)
                
    if legend:
        fig.legend(handles=legend_patches, loc='upper center', ncol=len(methods), fontsize='x-large', bbox_to_anchor=(0.5, 1.15))
        
    plt.tight_layout()
    plt.savefig(f'figure/alternate_nodes={num_nodes}.pdf', bbox_inches='tight')
    plt.show()

    