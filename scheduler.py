import sys
sys.dont_write_bytecode = True
import time
import random
import argparse
import asyncio
from typing import List
from producer import Producer, Task


# DeviceQueue representing a GPU device's queue
class DeviceQueue:
    def __init__(self, cuda_id: int):
        self.queue = asyncio.Queue()
        self.cuda_id = cuda_id
        self.stop_signal = asyncio.Event()

    async def add_task(self, task: Task):
        await self.queue.put(task)
        
    async def process_tasks(self):
        # while not self.stop_signal.is_set():
        #     try:
        #         task = await self.queue.get()
        #         await self.process_task(task)
        #         self.queue.task_done()
        #     except asyncio.TimeoutError:
        #         continue
        # print(f"Stopped processing on CUDA device {self.cuda_id}")
        while True:
            try:
                task = await asyncio.wait_for(self.queue.get(), timeout=1)
                await self.process_task(task)
                self.queue.task_done()
            except asyncio.TimeoutError:
                if self.stop_signal.is_set() and self.queue.empty():
                    break  # Exit the loop if stop signal is set and queue is empty

        print(f"Stopped processing on CUDA device {self.cuda_id}")

    async def process_task(self, task: Task):
        print(f"Processing task on CUDA device {self.cuda_id} at time {time.time()}")
        # Inference / training
        ############################################################
        
        ############################################################
        task.processed = True
        
        

class Node:
    """
    Suppose each node (server) has 2 GPUs, each having a queue.
    """
    def __init__(self, id: int, cuda_devices: List[int]):
        self.id = id
        self.devices = [DeviceQueue(cuda_id) for cuda_id in cuda_devices]
        
    def schedule_device(self):
        """
        Schedule a device by load balancing.
        """
        return min(self.devices, key=lambda device: device.queue.qsize())
        
    async def add_task(self, task):
        device = self.schedule_device()
        await device.add_task(task)
        
        

class GlobalScheduler:
    
    def __init__(
        self, 
        task_queue: asyncio.Queue, 
        nodes: List[Node],
        task_complete_flag: asyncio.Event,
    ):
        self.task_queue = task_queue
        self.nodes = nodes
        self.task_complete_flag = task_complete_flag
        
    async def schedule(self):
        while True:
            # Set terminate condition
            if self.task_complete_flag.is_set() and self.task_queue.empty():
                print("Global scheduler finished scheduling tasks")
                # Signal nodes to stop processing
                for node in self.nodes:
                    for device in node.devices:
                        device.stop_signal.set()
                break
            
            task = await self.task_queue.get()
            node = random.choice(self.nodes)
            await node.add_task(task)
            print(f"Task scheduled to node {node.id} at time {time.time()}")
            print(f"Task queue size: {self.task_queue.qsize()}")
            
            
async def main(args):
    task_queue = asyncio.Queue()
    task_complete_flag = asyncio.Event()
    node1 = Node(id=1, cuda_devices=[0, 1])
    node2 = Node(id=2, cuda_devices=[2, 3])
    nodes = [node1, node2]
    
    producer = Producer(task_queue, args, task_complete_flag)
    scheduler = GlobalScheduler(task_queue, nodes, task_complete_flag)

    # Start all components as asyncio tasks
    producer_task = asyncio.create_task(producer.produce())
    scheduler_task = asyncio.create_task(scheduler.schedule())
    device_tasks = [asyncio.create_task(device.process_tasks()) for node in nodes for device in node.devices]
    
    # Run all tasks until complete
    await asyncio.gather(producer_task, scheduler_task, *device_tasks)
        
        

if __name__ == "__main__":
    
    # Test the producer
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_lambda', type=float, default=100, help='Average number of tasks produced per minute')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], 
                        default='random', help='workload setting')
    args = parser.parse_args()
    
    asyncio.run(main(args))
    
    
    
    