import sys
sys.dont_write_bytecode = True
import asyncio
import random
import time
import argparse
from tqdm import tqdm
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from dataset import get_data, SentencePairDataset


class Task:
    """
    This class encapsulates a task to be processed by the system.
    query (str): the query batch to be processed
    timestamp (float): the time at which the task was generated
    feedback (str): the feedback for the query (optional) from the user
    """
    def __init__(self, query: Tensor, timestamp: float, feedback: Tensor = None):
        self.query = query
        self.timestamp = timestamp
        self.feedback = feedback
        self.processed = False


class Producer:
    def __init__(
        self, 
        task_queue: asyncio.Queue, 
        args: argparse.Namespace,
        task_complete_flag: asyncio.Event,
    ):
        self.task_queue = task_queue
        self.rate_lambda = args.rate_lambda
        
        train_data, val_data, test_data, self.vocab = get_data(setting=args.setting)
        
        def collate_batch(batch):
            # 'batch' is a list of tuples with (sequence, target)
            batch_data, batch_target = zip(*batch)
            combined_list = batch_data + batch_target
            # Dynamically pad the batch
            padded = pad_sequence(combined_list, batch_first=True, padding_value=self.vocab['<pad>'])
            padded_data = padded[:len(batch_data)]
            padded_target = padded[len(batch_data):]
            return padded_data, padded_target.view(-1)
        
        # dataset = SentencePairDataset(train_data)
        self.dataset = SentencePairDataset(test_data)
        # Sample
        if args.n_samples > 0:
            self.dataset = Subset(self.dataset, random.sample(range(len(self.dataset)), args.n_samples))
        
        self.dataloader = DataLoader(self.dataset, batch_size=args.bptt, collate_fn=collate_batch)
        
        self.task_complete_flag = task_complete_flag
        
    async def produce(self):
        # Produce using the dataset
        # while True:
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            # print("batch: ", batch)
            print(f"query shape: {batch[0].shape}, target shape: {batch[1].shape}")
            await asyncio.sleep(random.expovariate(self.rate_lambda))
            # 5% of the time, produce a task with feedback
            if random.random() < 0.05:
                task = Task(query=batch[0], timestamp=time.time(), feedback=batch[1])
            else:
                task = Task(query=batch[0], timestamp=time.time())
            await self.task_queue.put(task)
            # print(f"Produced a task {task} at time {task.timestamp}")
        self.task_complete_flag.set()
        print("Producer finished producing tasks")
            


if __name__ == "__main__":
    
    # Test the producer
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate_lambda', type=float, default=60, help='Average number of tasks produced per minute')
    parser.add_argument('--setting', type=str, choices=['identical','random', 'increasing', 'decreasing'], 
                        default='random', help='workload setting')
    parser.add_argument('--bptt', type=int, default=5, help='batch size')
    args = parser.parse_args()
    
    task_queue = asyncio.Queue()
    producer = Producer(task_queue, args)
    asyncio.run(producer.produce())
    
    