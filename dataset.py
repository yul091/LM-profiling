import random
import torch
from torch import nn, Tensor
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.utils.data import Dataset


# Load and batch data 
def data_process(vocab, tokenizer, raw_text_iter):
  data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(
    data: Tensor, 
    bsz: int, 
    setting: str, 
    min_len: int = 10, 
    max_len: int = 128,
):
    # # Divide the dataset into ``bsz`` parts.
    # nbatch = data.size(0) // bsz
    # # Trim off any extra elements that wouldn't cleanly fit (remainders).
    # data = data.narrow(0, 0, nbatch * bsz) # tensor(#tokens)
    # # Evenly divide the data across the ``bsz` batches.
    # data = data.view(bsz, -1).t().contiguous() # (N X bsz)
    # return data.cuda(0)
    
    # List to store sentence tensors
    sentences = []
    data_len = data.size(0)
    i = 0
    # Loop through the 'data' to create sentences with random lengths
    while i < data_len:
        # Generate a random length for the sentence
        if setting != 'identical':
            rand_length = min(random.randint(min_len, max_len), data_len - i)
        else:
            rand_length = bsz
        # Slice the data to get a "sentence" of 'rand_length'
        sentence = data[i:i+rand_length]
        # Add the tensor representing the sentence to the list
        sentences.append(sentence)
        # Increment 'i' to the next starting point
        i += rand_length
    # return sentences
    if setting == 'increasing':
        sentences.sort(key=lambda x: x.size(0))
    elif setting == 'decreasing':
        sentences.sort(key=lambda x: x.size(0), reverse=True)
    return sentences
        

def get_data(
    batch_size: int = 20, 
    eval_batch_size: int = 20, 
    setting: str = 'identical', 
    min_len: int = 10, 
    max_len: int = 128,
):
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(vocab, tokenizer, train_iter) # tensor(#train_tokens)
    val_data = data_process(vocab, tokenizer, val_iter) # tensor(#val_tokens)
    test_data = data_process(vocab, tokenizer, test_iter) # tensor(#test_tokens)

    train_data = batchify(train_data, batch_size, setting, min_len, max_len)
    val_data = batchify(val_data, eval_batch_size, setting, min_len, max_len)
    test_data = batchify(test_data, eval_batch_size, setting, min_len, max_len)
    
    return train_data, val_data, test_data, vocab


class SentencePairDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list (list of Tensors): A list where each element is a tensor corresponding to a sentence.
        """
        self.data_list = data_list

    def __len__(self):
        # We return the length minus one because we are creating pairs of sentences
        return len(self.data_list) - 1

    def __getitem__(self, idx):
        # Return the current sentence and the next sentence as the data and target, respectively.
        data = self.data_list[idx]
        target = self.data_list[idx + 1]
        return data, target