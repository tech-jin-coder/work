import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from ae import AutoEncoder


BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 150                           # 记忆库容量
N_STATES = 1000 + 1024 * 2                      # N_STATES = state(after encoder) , hx , cx


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

class Net(nn.Module):
    def __init__(self,action_space,model,in_channels = 3):
        super(Net,self).__init__()
        self.in_channels = in_channels
        self.action_space = action_space
        self.autoencoder = model
        for child in self.autoencoder.children():
            for param in child.parameters():
                param.requires_grad = False


        self.conv1 = nn.Conv2d(in_channels,32,5,stride=1,padding=2)
        self.maxp1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(32,32,5,stride=1,padding=1)
        self.maxp2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(32,64,4,stride=1,padding=1)
        self.maxp3 = nn.MaxPool2d(2,2)

        self.conv4 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.maxp4 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(64,64,3,stride=1,padding=1)
        self.maxp5 = nn.MaxPool2d(2,2)

        self.lstm = nn.LSTMCell(64*7*7, 1024)
        self.out = nn.Linear(1024,256*256)
        
        self.fc1 = nn.Linear(1000,128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out_action = nn.Linear(128, action_space)
        self.out_action.weight.data.normal_(0, 0.1)


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv5.weight.data.mul_(relu_gain)
        self.out.weight.data.normal_(0, 0.1)
       

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self,input):
        input,(hx,cx) = input

        actin_x = self.fc1(input)

        actin_x = F.relu(actin_x)

        actions_value = self.out_action(actin_x)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        
        input = self.autoencoder.decode(input).type(torch.IntTensor)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        
        if action == 0:   #click
            input = torch.where(((input/17)&2)==2,ones,zeros)
        if action == 1:   #long-click
            input = torch.where(((input/17)&4)==4,ones,zeros)
        if action == 2:   #check
            input = torch.where(((input/17)&1)==1,ones,zeros)
        if action == 3:   #left_scroll
            input = torch.where(((input/17)&8)==8,ones,zeros)
        if action == 4:   #right_scroll
            input = torch.where(((input/17)&8)==8,ones,zeros)
        if action == 5:   #up_scroll
            input = torch.where(((input/17)&8)==8,ones,zeros)
        if action == 6:   #down_scroll
            input = torch.where(((input/17)&8)==8,ones,zeros)
        input = input.type(torch.cuda.FloatTensor).detach()
        x = F.relu(self.maxp1(self.conv1(input)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = F.relu(self.maxp5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))

        x = hx
        
        return actions_value,self.out(x),(hx,cx)

class DDQN(object):
    def __init__(self,action_space,model,device,in_channels = 3): 
        self.action_space = action_space
        self.autoencoder = model
        self.device = device
        self.in_channels = in_channels
        self.eval_net, self.target_net = Net(action_space,model,in_channels), Net(action_space,model,in_channels) 
        self.eval_net.to(self.device)
        self.target_net.to(self.device)
        self.memory_counter = 0  
        self.MEMORY_CAPACITY = MEMORY_CAPACITY                                            # for storing memory
        self.memory_s = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 4))
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 4))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss().to(self.device)                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
    
    def  choose_action(self,state):
        tmp = np.random.uniform()
        if tmp < EPSILON:
            action_value,location,(hx,cx) = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
            location = torch.max(location, 1)[1].data.cpu().numpy()[0]
            location = (location / 256, location % 256)
        else: 
            action = np.random.randint(0, self.action_space)
            location,(hx,cx) = state
            location = torch.max(location, 1)[1].data.cpu().numpy()[0]
            location = (location / 256, location % 256)
        if tmp > 0.99:
            action = 7
        return action,location,(hx,cx)
    
    def store_transition(self, s, a, r, s_):
        print(s.size())
        print(a.size())
        print(r.size())
        print(s_.size())
        transition = np.hstack((s, [a, r], s_))
        print(transition.size())
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+3])
        b_aciton =  b_a[0].astype(int)
        b_location = (b_a[1].astpye(int),b_a[2].astpye(int)) 
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+3:N_STATES+4])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval_action , q_eval_location , (q_eval_hx , q_eval_cx) = self.eval_net(b_s)
        q_eval_action = q_eval_action.gather(1,b_aciton)
        q_eval_location = q_eval_location.gather(1,b_location[0] * 256 +b_location[1])
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next_action , q_next_location , (q_next_hx , q_next_cx) = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target_action = b_r + GAMMA * q_next_action.max(1)[0].view(BATCH_SIZE, 1)
        q_target_location = b_r + GAMMA * q_next_action.max(1)[0].view(BATCH_SIZE, 1)
        # q_next_action.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss_a = self.loss_func(q_eval_action, q_target_action)
        loss_loc = self.loss_func(q_eval_location, q_target_location)
        loss = loss_a + loss_loc
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                              # 清空上一步的残余更新参数值
        loss.backward()                                                         # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                                   # 更新评估网络的所有参数
