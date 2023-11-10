import sys
sys.dont_write_bytecode = True
import asyncio
import random
import time
import argparse
from tqdm import tqdm
from torch import Tensor

from dataset import get_data, SentencePairDataset


class Task:
    """
    This class encapsulates a task to be processed by the system.
    query (str): the query to be processed
    timestamp (float): the time at which the task was generated
    feedback (str): the feedback for the query (optional) from the user
    """
    def __init__(self, query: Tensor, timestamp: float, feedback: Tensor = None):
        self.query = query
        self.timestamp = timestamp
        self.feedback = feedback
        self.processed = False


class Producer:
    def __init__(self, task_queue: asyncio.Queue(), args: argparse.Namespace):
        self.task_queue = task_queue
        self.rate_lambda = args.rate_lambda
        train_data, val_data, test_data, vocab = get_data(setting=args.setting)
        # self.dataset = SentencePairDataset(train_data)
        self.dataset = SentencePairDataset(test_data)
        
    async def produce(self):
        # Produce using the dataset
        # while True:
        for i, instance in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            await asyncio.sleep(random.expovariate(self.rate_lambda))
            # 5% of the time, produce a task with feedback
            if random.random() < 0.05:
                task = Task(query=instance[0], timestamp=time.time(), feedback=instance[1])
            else:
                task = Task(query=instance[0], timestamp=time.time())
            await self.task_queue.put(task)
            # print(f"Produced a task {task} at time {task.timestamp}")
            
            
if __name__ == "__main__":
    
    # Test the producer
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_lambda', type=float, default=100, help='Average number of tasks produced per minute')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], 
                        default='identical', help='workload setting')
    args = parser.parse_args()
    
    task_queue = asyncio.Queue()
    producer = Producer(task_queue, args)
    asyncio.run(producer.produce())
    
    