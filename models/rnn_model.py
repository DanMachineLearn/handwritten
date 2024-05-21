# -*- encoding: utf-8 -*-
'''
@File		:	rnn_model.py
@Time		:	2024/05/21 15:09:59
@Author		:	dan
@Description:	RNN模型
'''

import torch
import torch.nn as nn
import torch.optim as optim

class HandwrittenRNN(nn.Module):
    def __init__(self, output_size, input_size = 6, hidden_size = 1024, num_layers=1):
        super(HandwrittenRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 通过RNN层
        out, _ = self.rnn(x, h0)
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out



def main():
    # 定义超参数
    input_size = 6   # 输入特征数
    hidden_size = 20  # 隐藏层特征数
    output_size = 1024   # 输出特征数
    num_layers = 2    # RNN层数
    learning_rate = 0.001
    num_epochs = 100

    
    # 创建模型
    model = HandwrittenRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 示例输入数据
    # 形状：(批量大小, 序列长度, 输入特征数)
    x_train = torch.randn(32, 5, input_size)  # 批量大小为32，序列长度为5
    y_train = torch.randn(32, output_size, dtype=torch.long)    # 批量大小为32

    # 训练模型
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training completed.")

if __name__ == '__main__':
    main()

