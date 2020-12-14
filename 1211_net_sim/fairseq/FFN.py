from torch import nn
# Neural Network Model (1 hidden layer)
class FFNet(nn.Module):
    #初始化网络结构
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #输入层，线性（liner）关系
        self.relu = nn.ReLU()#隐藏层，使用ReLU函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  #输出层，线性（liner）关系
    #forword 参数传递函数，网络中数据的流动
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
