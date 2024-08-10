import torch
from torch import nn
class TinyVGG(nn.Module):
  def __init__(self,input,output,hidden_unit):
    super().__init__()
    self.conv_layer1=nn.Sequential(
        nn.Conv2d(
            in_channels=input,
            out_channels=hidden_unit,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_unit,
            out_channels=hidden_unit,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_layer2=nn.Sequential(
        nn.Conv2d(
            in_channels=hidden_unit,
            out_channels=hidden_unit,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=hidden_unit,
            out_channels=hidden_unit,
            kernel_size=3,
            padding=1,
            stride=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=hidden_unit*16*16,
            out_features=output
        )
    )

  def forward(self,x):
    x=self.conv_layer1(x)

    x=self.conv_layer2(x)

    x=self.classifier(x)

    return x
