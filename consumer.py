import sys
sys.dont_write_bytecode = True
import asyncio
from typing import List
import torch
import torch.nn as nn
from utils import record_time
from producer import Task


# DeviceQueue representing a GPU device's queue
class DeviceQueue:
    def __init__(self, cuda_id: int):
        self.queue = asyncio.Queue()
        self.cuda_id = cuda_id
        self.stop_signal = asyncio.Event()

    async def add_task(self, task: Task):
        await self.queue.put(task)

    async def process_task(self, task: Task, timing_info: dict, stage: nn.Module, next_cuda_id: int = None):
        # Inference
        #####################################################################################################################
        with torch.no_grad():
            # Record the start time of the stage on this GPU
            record_time(self.cuda_id, 'start', timing_info, verbose=True)
            # First put query into cuda device
            task.query = task.query.cuda(self.cuda_id)
            output = stage(task.query)
            record_time(self.cuda_id, 'end', timing_info, verbose=True)
            if next_cuda_id:
                output = output.cuda(next_cuda_id)  # Move output to the next stage's device
            task.query = output
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
        
    async def add_task(self, task):
        # device = self.schedule_device()
        await self.devices[0].add_task(task) # always add input data on the first device
        
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
            # try:
            # task: Task = await asyncio.wait_for(device_queue_in.get(), timeout=1)
            task: Task = await device_queue_in.get()
            if task is None:  # None is sent as a signal to shut down
                await device_queue_out.put(None)
                break
            await self.devices[device_id].process_task(task, timing_info, stage, next_device_id)
            await device_queue_out.put(task)
            # except:
            #     if self.devices[device_id].stop_signal.is_set() and device_queue_in.empty():
            #         break
            
        
        
            
            
