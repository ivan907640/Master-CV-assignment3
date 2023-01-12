import torch.nn as nn

# convolutional layer 1
conv_layer1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5)),
    nn.ReLU(),
)

# convolutional layer 2 
conv_layer2 = nn.Sequential(
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
    nn.ReLU(),
)

# convolutional layer 3
conv_layer3 = nn.Sequential(
    nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1,1)),
    nn.ReLU(),
)

# fully connected layer 1
fc_layer1 = nn.Sequential(
    nn.Linear(in_features=8*5*5, out_features=64),
    nn.ReLU(),
)

# fully connected layer 2
fc_layer2 = nn.Sequential(
    nn.Linear(in_features=64, out_features=10)
)



CNN = nn.Sequential(
    conv_layer1,                      #24*24*16
    nn.MaxPool2d(kernel_size=(2,2)),  #12*12*16
    conv_layer2,                      #10*10*32
    nn.MaxPool2d(kernel_size=(2,2)),  #5*5*32
    conv_layer3,                      #5*5*8
    nn.Flatten(),                     #200
    fc_layer1,
    fc_layer2
)
