import numpy as np

BW = 20  # 带宽
NFFT = int(BW * 3.2)
GESTURE_NUM = 3  # 手势数
TRAIN_SIZE = 210  # 训练集大小
PCAP_SIZE = 150  # 每个pcap包含的CSI数组个数

train_data = np.zeros((TRAIN_SIZE, PCAP_SIZE, NFFT))
train_result = np.zeros((TRAIN_SIZE, GESTURE_NUM))

for i in range(1, 70 + 1):
    # 读取npy文件
    temp_T1 = np.load('data/T1/T1_' + str(i) + '.npy')
    temp_T2 = np.load('data/T2/T2_' + str(i) + '.npy')
    temp_T3 = np.load('data/T3/T3_' + str(i) + '.npy')

    # CSI求模&&毛刺处理
    for j in range(PCAP_SIZE):
        for k in range(NFFT):
            temp_T1[j][k] = abs(temp_T1[j][k])
            temp_T2[j][k] = abs(temp_T2[j][k])
            temp_T3[j][k] = abs(temp_T3[j][k])
            if k == 0 or k == 29 or k == 30 or k == 31 or k == 32 or k == 33 or k == 34 or k == 35:
                temp_T1[j][k] = 0
                temp_T2[j][k] = 0
                temp_T3[j][k] = 0

    # 加入学习队列
    # temp_T1.dtype = float
    # temp_T2.dtype = float
    # temp_T3.dtype = float
    train_data[i - 1] = temp_T1[0:PCAP_SIZE, :]
    train_data[70 + i - 1] = temp_T2[0:PCAP_SIZE, :]
    train_data[140 + i - 1] = temp_T3[0:PCAP_SIZE, :]
    train_result[i - 1] = [1, 0, 0]
    train_result[70 + i - 1] = [0, 1, 0]
    train_result[140 + i - 1] = [0, 0, 1]

# 训练数据归一化
cnt = 0
for i in range(TRAIN_SIZE):
    for j in range(PCAP_SIZE):
        #print(train_data[i][j])
        CSI_max = train_data[i][j].max()
        if CSI_max == 0:
            cnt = cnt + 1
        for k in range(NFFT):
            # if train_data[i][j][k] == 0:
            #     train_data[i][j][k] = 1
            train_data[i][j][k] = train_data[i][j][k]/CSI_max






import torch

# N is batch size; INPUT_SIZE is input dimension;
# HIDDEN_WIDTH is the width of hidden dimension; OUTPUT_SIZE is output dimension;
# VAL_SIZE is validation size.
N = 210
INPUT_SIZE = 64*150
HIDDEN_WIDTH = 50
OUTPUT_SIZE = 3
VAL_SIZE = 10
epochs = 500


class Net(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_WIDTH, OUTPUT_SIZE):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(INPUT_SIZE, HIDDEN_WIDTH)
        self.linear2 = torch.nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH)
        self.linear3 = torch.nn.Linear(HIDDEN_WIDTH, OUTPUT_SIZE)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        # use dropout() to prevent over learning
        x = self.dropout(torch.nn.functional.relu(self.linear1(x)))
        x = self.dropout(torch.nn.functional.relu(self.linear2(x)))
        # the output dimension shouldn't use dropout()
        y_pred = torch.nn.functional.softmax(self.linear3(x), dim=1)
        return y_pred



# Create random Tensors to hold inputs and outputs
x = torch.from_numpy(train_data[0:59])
y = torch.from_numpy(train_result[0:59])
x = x.view(59, -1)

test_x = torch.from_numpy(train_data[60:69])
test_y = torch.from_numpy(train_result[60:69])
test_x = test_x.view(10, -1)
# Construct our model by instantiating the class defined above
model = Net(INPUT_SIZE, HIDDEN_WIDTH, OUTPUT_SIZE)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=10)
for t in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x.float())
    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t + 1, '针对训练集的损失', loss.item())
        with torch.no_grad():
            y_pred = model(test_x)
            test_loss = criterion(y_pred, test_y)
            total = torch.tensor(VAL_SIZE, dtype=float)
            bingo = (y_pred == test_y.squeeze(1)).sum()
            print('针对测试集的损失', test_loss.item(), '准确率', (bingo/total).item())
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
