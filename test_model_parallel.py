import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x



class MLPParallel(nn.Module):
    def __init__(
        self, 
        input_size: int = 256,
        output_size: int = 256, 
        resid_pdrop: float = 0.5,
    ):
        super().__init__()
        self.c_fc = Conv1D(input_size, output_size).cuda(0)
        self.act = ReLUSquaredActivation().cuda(1)
        self.c_proj = Conv1D(output_size, input_size).cuda(2)
        self.dropout = nn.Dropout(resid_pdrop).cuda(3)
        
    # def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
    #     hidden_states = self.c_fc(hidden_states)
    #     hidden_states = self.act(hidden_states)
    #     hidden_states = self.c_proj(hidden_states)
    #     hidden_states = self.dropout(hidden_states)
    #     return hidden_states
    
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        x = hidden_states.cuda(0)
        x = self.c_fc(x)
        x = x.cuda(1)
        x = self.act(x)
        x = x.cuda(2)
        x = self.c_proj(x)
        x = x.cuda(3)
        x = self.dropout(x)
        return x
    
    
    
if __name__ == "__main__":
    
    inputs = torch.rand(8, 256).cuda(0)
    labels = torch.randint_like(inputs, 10).contiguous().cuda(3)
    
    model = MLPParallel()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    outputs = model(inputs)
    print(outputs.shape)
    
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(outputs, labels)
    
    loss.backward()
    print(model.c_fc.weight)
    print(model.c_fc.weight.grad)
    
    optimizer.step()
    optimizer.zero_grad()
    print(model.c_fc.weight)
    print(model.c_fc.weight.grad)
    
    
    
    
    
    
    
