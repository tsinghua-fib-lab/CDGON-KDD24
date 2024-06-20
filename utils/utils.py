import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data,mean= 0,std=0.1)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
