# -*- encoding: utf-8 -*-
'''
@File		:	gru_model.py
@Time		:	2024/05/15 10:18:52
@Author		:	dan
@Description:	GRU 模型
'''
from torch import nn
import torch

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 GRU
        out, _ = self.gru(x, h0)
        
        # 解码隐藏状态的输出
        out = self.fc(out)
        return out

def main():
    input_size = 1
    hidden_size = 50
    output_size = 1
    num_layers = 1

    model = GRUModel(input_size, hidden_size, output_size, num_layers)
    print(model)

if __name__ == '__main__':
    main()


