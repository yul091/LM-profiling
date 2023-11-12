import sys
sys.dont_write_bytecode = True
import time
import random
import asyncio
from typing import List
from consumer import Node


class GlobalScheduler:
    
    def __init__(
        self, 
        task_queue: asyncio.Queue, 
        nodes: List[Node],
    ):
        self.task_queue = task_queue
        self.nodes = nodes
        
    async def schedule(self):
        while True:
            # # Set terminate condition
            # if self.task_complete_flag.is_set() and self.task_queue.empty():
            #     print("Global scheduler finished scheduling tasks")
            #     # Signal nodes to stop processing
            #     for node in self.nodes:
            #         for device in node.devices:
            #             device.stop_signal.set()
            #     break
            
            task = await self.task_queue.get()
            if task is None:
                print("Global scheduler finished scheduling tasks")
                # Signal nodes to stop processing
                for node in self.nodes:
                    for device in node.devices:
                        device.stop_signal.set()
                break
            
            node = random.choice(self.nodes)
            await node.add_task(task)
            print(f"Task scheduled to node {node.id} at time {time.time()}")
            print(f"Task queue size: {self.task_queue.qsize()}")
            
            

        
        


    
    
    
    