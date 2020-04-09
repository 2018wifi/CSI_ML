import torch
import torch.nn.functional as functional
import numpy as np
import random

# N is batch size; INPUT_SIZE is input dimension;
# HIDDEN_WIDTH is the width of hidden dimension; OUTPUT_SIZE is output dimension;
# VAL_SIZE is validation size.
BW = 20  # 带宽
NFFT = int(BW * 3.2)
PCAP_SIZE = 150  # 每个pcap包含的CSI数组个数
T_NUM = 3
T_SIZE = 70


INPUT_SIZE = PCAP_SIZE * NFFT
HIDDEN_WIDTH = 50
VAL_SIZE = 20
OUTPUT_SIZE = T_NUM
epochs = 50


class Net(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_WIDTH, OUTPUT_SIZE):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(INPUT_SIZE, HIDDEN_WIDTH)
        self.linear2 = torch.nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH)
        self.linear3 = torch.nn.Linear(HIDDEN_WIDTH, OUTPUT_SIZE)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        # use dropout() to prevent over learning
        x = self.dropout(functional.relu(self.linear1(x)))
        x = self.dropout(functional.relu(self.linear2(x)))
        # the output dimension shouldn't use dropout()
        y_pred = functional.softmax(self.linear3(x), dim=1)
        return y_pred


x = torch.zeros((T_NUM * T_SIZE, INPUT_SIZE))
y = torch.zeros((T_NUM * T_SIZE, OUTPUT_SIZE))
cnt = 0
for i in range(T_NUM):
    for j in range(T_SIZE):
        temp_y = torch.zeros(T_NUM)
        temp_y[i] = 1
        # read .npy file
        file = 'data/T' + str(i+1) + '/T' + str(i+1) + '_' + str(j+1) + '.npy'
        temp_x = np.load(file)
        for k in range(PCAP_SIZE):
            temp_x[k] = abs(temp_x[k])
        temp_x = torch.from_numpy(temp_x.astype('float64'))[0:PCAP_SIZE]
        # clean and normalize data
        for k in [0, 29, 30, 31, 32, 33, 34, 35]:
            temp_x[:, k] = 0
        for k in range(temp_x.shape[0]):
            CSI_max = temp_x[k].max()
            for p in range(temp_x.shape[1]):
                temp_x[k][p] = temp_x[k][p] / CSI_max
        # flatten the 3D data
        temp_x = temp_x.view((-1, ))
        # concatenates to x and y
        x[cnt] = temp_x
        y[cnt] = temp_y
        cnt += 1
        print(i, j)

# Extract test batch
test_x = torch.zeros((VAL_SIZE, INPUT_SIZE))
test_y = torch.zeros((VAL_SIZE, OUTPUT_SIZE))
for i in range(VAL_SIZE):
    n = random.randint(0, x.shape[0] - 1)
    print(n)
    test_x[i] = x[n]
    test_y[i] = y[n]
    x = torch.cat((x[:n], x[n+1:]))
    y = torch.cat((y[:n], y[n+1:]))

# Construct our model by instantiating the class defined above
model = Net(INPUT_SIZE, HIDDEN_WIDTH, OUTPUT_SIZE)

# Construct a loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
for t in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x.float())
    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t + 1, '针对训练集的损失', loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
