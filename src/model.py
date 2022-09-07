import torch 
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv_layer_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.conv_layer_3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=(0,0))
        self.linear_layer_1 = nn.Linear(in_features=120, out_features=84)
        self.linear_layer_2 = nn.Linear(in_features=84, out_features=7)
        self.pooling_layer = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.tanh(x)
        x = self.pooling_layer(x)
        
        x = self.conv_layer_2(x)
        x = self.tanh(x)
        x = self.pooling_layer(x)
        
        x = self.conv_layer_3(x)
        x = self.tanh(x)
        
        x = x.reshape((x.shape[0], -1))
        x = self.linear_layer_1(x)
        x = self.tanh(x)
        
        x = self.linear_layer_2(x)
        
        return x

model = LeNet()