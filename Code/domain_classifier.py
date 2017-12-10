import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_1_size, hidden_layer_2_size):
        super(DomainClassifier, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_layer_1_size)
        self.hidden2 = nn.Linear(hidden_layer_1_size, hidden_layer_2_size)
        self.output = nn.Linear(hidden_layer_2_size, 2)

    def forward(self, inputs):
        x = self.hidden1(inputs)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        return x