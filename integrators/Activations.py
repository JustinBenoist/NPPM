import torch
import torch.nn as nn

class ScaledTanh(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input):
        res = self.factor * torch.tanh(input)
        return res
    
class ScaledSigmoid(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input):
        res = self.factor * torch.sigmoid(input)
        return res
    
class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.exp(input) + 1e-7
    
class SinAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)
    
class CeluAct(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return torch.celu(input) + 1.00001

class Absolute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.abs(input)
    
class ExponentialNegative(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return torch.exp(-input)

class NullAct(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input

class SqrtReLU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return torch.sqrt(torch.relu(input)) + 0.001

class Square(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return torch.square(input)

class EluAct(torch.nn.ELU):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return super().forward(input - 2) + 1.001