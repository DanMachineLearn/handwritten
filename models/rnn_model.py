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
    def __init__(self, output_size, input_size=6, hidden_size=500, num_layers=1):
        super(HandwrittenRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=200)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=200, out_features=output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 只使用最后一个时间步的输出
        out = self.fc1(out[:, -1, :])
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def main():
    # 定义超参数
    input_size = 6   # 输入特征数
    output_size = 1024   # 输出特征数
    learning_rate = 0.001
    num_epochs = 100

    
    # 创建模型
    model = HandwrittenRNN(output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 示例输入数据
    # 形状：(批量大小, 序列长度, 输入特征数)
    x_train = torch.randn(32, 5, input_size)  # 批量大小为32，序列长度为5
    y_train = torch.randn(32, output_size)    # 批量大小为32

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

