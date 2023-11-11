import sys
sys.dont_write_bytecode = True
import time
import argparse
import asyncio
from typing import List
import torch
import torch.nn as nn

from utils import record_time
from producer import Producer, Task



# DeviceQueue representing a GPU device's queue
class DeviceQueue:
    def __init__(self, cuda_id: int):
        self.queue = asyncio.Queue()
        self.cuda_id = cuda_id
        self.stop_signal = asyncio.Event()

    async def add_task(self, task: Task):
        await self.queue.put(task)
        
    # async def process_tasks(self, stage: nn.Module, next_cuda_id: int = None):
    #     # while not self.stop_signal.is_set():
    #     #     try:
    #     #         task = await self.queue.get()
    #     #         await self.process_task(task)
    #     #         self.queue.task_done()
    #     #     except asyncio.TimeoutError:
    #     #         continue
    #     # print(f"Stopped processing on CUDA device {self.cuda_id}")
    #     while True:
    #         try:
    #             task = await asyncio.wait_for(self.queue.get(), timeout=1)
    #             await self.process_task(task, stage, next_cuda_id)
    #             self.queue.task_done()
    #         except asyncio.TimeoutError:
    #             if self.stop_signal.is_set() and self.queue.empty():
    #                 break  # Exit the loop if stop signal is set and queue is empty
    #     print(f"Stopped processing on CUDA device {self.cuda_id}")

    async def process_task(self, task: Task, timing_info: dict, stage: nn.Module, next_cuda_id: int = None):
        print(f"Processing task on CUDA device {self.cuda_id} at time {time.time()}")
        # Inference
        #####################################################################################################################
        with torch.no_grad():
            # Record the start time of the stage on this GPU
            record_time(self.cuda_id, 'start', timing_info)
            # print("self cuda id: ", self.cuda_id, "stage device: ", stage.device, "task query device: ", task.query.device)
            # First put query into cuda device
            task.query = task.query.cuda(self.cuda_id)
            output = stage(task.query)
            record_time(self.cuda_id, 'end', timing_info)
            if next_cuda_id:
                output = output.cuda(next_cuda_id)  # Move output to the next stage's device
            task.query = output
            print(f"task query shape: {task.query.shape}")
        #####################################################################################################################
        task.processed = True
        
        

class Node:
    """
    Suppose each node (server) has 2 GPUs, each having a queue.
    """
    def __init__(self, id: int, cuda_devices: List[int], criterion: nn.CrossEntropyLoss):
        self.id = id
        self.devices = [DeviceQueue(cuda_id) for cuda_id in cuda_devices]
        self.criterion = criterion
        
    # def schedule_device(self):
    #     """
    #     Schedule a device by load balancing.
    #     """
    #     return min(self.devices, key=lambda device: device.queue.qsize())
        
    async def add_task(self, task):
        # device = self.schedule_device()
        await self.devices[0].add_task(task) # always add input data on the first device
        
    async def compute_loss(self, queue: asyncio.Queue, ntokens: int, total_size: int):
        total_loss = 0.
        while True:
            task: Task = await queue.get() # (B X T X C) and (B*T)
            if task.feedback is None:
                break
            print(f"query shape: {task.query.shape}, target shape: {task.feedback.shape}")
            output_flat = task.query.contiguous().view(-1, ntokens) # (B*T) X C
            total_loss += task.feedback.size(0) * self.criterion(output_flat, task.feedback.to(task.query.device)).item() 
        return total_loss / total_size
        
    async def coroutine_inference_stage(
        self,
        stage: nn.Module, 
        device_queue_in: asyncio.Queue, 
        device_queue_out: asyncio.Queue, 
        device_id: int, 
        timing_info: dict,
        next_device_id: int = None,
    ):
        while True:
            try:
                task: Task = await asyncio.wait_for(device_queue_in.get(), timeout=1)
                # if task is None:  # None is sent as a signal to shut down
                #     await device_queue_out.put((None, None))
                #     break
                await self.devices[device_id].process_task(task, timing_info, stage, next_device_id)
                await device_queue_out.put(task)
            except:
                if self.devices[device_id].stop_signal.is_set() and device_queue_in.empty():
                    break
            
    # async def coroutine_evaluate_stages(
    #     self,
    #     stages: List[nn.Module], 
    #     ntokens: int, 
    #     total_size: int, 
    #     timing_info: dict,
    # ):
    #     queues = [device.queue for device in self.devices] + [asyncio.Queue()]
        
    #     # Start stage coroutines
    #     stages_coros = [self.coroutine_inference_stage(
    #         stages[i], 
    #         queues[i],  # input queue (of data) for stage i
    #         queues[i + 1],  # output queue (of data) for stage i+1
    #         device=i, 
    #         timing_info=timing_info,
    #         next_device=i+1 if i < len(stages) - 1 else None,
    #     ) for i in range(len(stages))]
        
    #     # Start the consumer coroutine
    #     consumer_coro = self.compute_loss(queues[-1], ntokens, total_size)
        
        
            
            
