import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import sys
import copy

BATCH_SIZE = 16                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.8                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 50                        # 目标网络更新频率
MEMORY_CAPACITY = 500                           # 记忆库容量
N_STATES = 1000 + 1024 * 2                      # N_STATES = state(after encoder) , hx , cx
ones = torch.ones((1,256,256)).cuda()
zeros = torch.zeros((1,256,256)).cuda()

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
    def __init__(self,action_space,in_channels = 1):
        super(Net,self).__init__()
        self.in_channels = in_channels
        self.action_space = action_space



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

        self.lstm = nn.LSTMCell(64*7*7, 512)
        self.out = nn.Linear(512,256*256)
        
        self.fc1 = nn.Linear(1*256*256,128)
        # self.fc1.weight.data.normal_(0, 0.1)
        self.out_action = nn.Linear(128, action_space)
        # self.out_action.weight.data.normal_(0, 0.1)


        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv5.weight.data.mul_(relu_gain)
        # self.out.weight.data.normal_(0, 0.1)
       

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self,input):
        input,(hx,cx) = input
        input , action_possible = input
        input1 = input.view(input.size()[0],1*256*256).float()
        actin_x = self.fc1(input1)

        actin_x = torch.sigmoid(actin_x)
        acin_x = torch.sigmoid(self.out_action(actin_x))
        actions_value = torch.mul(acin_x,action_possible)
        action = torch.multinomial(actions_value, 1,replacement=True).data.cpu().numpy()
        if input.size()[0] == 1:
            if action[0][0] == 0:   #click
                input[0] = torch.where(((15-(input/17))&2)==2,ones.clone(),zeros.clone()).clone()
            if action[0][0] == 1:   #long-click
                input[0] = torch.where(((15-(input/17))&4)==4,ones.clone(),zeros.clone()).clone()
            if action[0][0] == 2:   #check
                input[0] = torch.where(((15-(input/17))&1)==1,ones.clone(),zeros.clone()).clone()
            if action[0][0] == 3:   #left_scroll
                input[0] = torch.where(((15-(input/17))&8)==8,ones.clone(),zeros.clone()).clone()
            if action[0][0] == 4:   #right_scroll
                input[0] = torch.where(((15-(input/17))&8)==8,ones.clone(),zeros.clone()).clone()
            if action[0][0] == 5:   #up_scroll
                input[0] = torch.where(((15-(input/17))&8)==8,ones.clone(),zeros.clone()).clone()
            if action[0][0] == 6:   #down_scroll
                input[0] = torch.where(((15-(input/17))&8)==8,ones.clone(),zeros.clone()).clone()
            if action[0][0] == 7:
                input[0] = ones.clone()
        else:
            for i in range(BATCH_SIZE):
                if action[i][0] == 0:   #click
                    input[i] = torch.where(((15-(input[i]/17))&2)==2,ones.clone(),zeros.clone()).clone()
                if action[i][0] == 1:   #long-click
                    input[i] = torch.where(((15-(input[i]/17))&4)==4,ones.clone(),zeros.clone()).clone()
                if action[i][0] == 2:   #check
                    input[i] = torch.where(((15-(input[i]/17))&1)==1,ones.clone(),zeros.clone()).clone()
                if action[i][0] == 3:   #left_scroll
                    input[i] = torch.where(((15-(input[i]/17))&8)==8,ones.clone(),zeros.clone()).clone()
                if action[i][0] == 4:   #right_scroll
                    input[i] = torch.where(((15-(input[i]/17))&8)==8,ones.clone(),zeros.clone()).clone()
                if action[i][0] == 5:   #up_scroll
                    input[i] = torch.where(((15-(input[i]/17))&8)==8,ones.clone(),zeros.clone()).clone()
                if action[i][0] == 6:   #down_scroll
                    input[i] = torch.where(((15-(input[i]/17))&8)==8,ones.clone(),zeros.clone()).clone()
                if action[i][0] == 7:
                    input[i] = ones.clone()
        input = input.type(torch.cuda.FloatTensor).detach()
        x = F.relu(self.maxp1(self.conv1(input)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = F.relu(self.maxp5(self.conv5(x)))
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        x = torch.sigmoid(self.out(x))
        y = input[:,0,:,:]
        y = y.view(-1,256*256)
        location_value = torch.mul(x,y)
        location = torch.multinomial(location_value, 1,replacement=True).data.cpu().numpy()
        #print("network")
        #print(action)
        #print(location)
        return action,actions_value,location,location_value,(hx,cx)

class DDQN(object):
    def __init__(self,action_space,device,appName,in_channels = 1): 
        self.action_space = action_space
        self.device = device
        self.appName = appName
        self.in_channels = in_channels
        self.eval_net, self.target_net = Net(action_space,in_channels), Net(action_space,in_channels) 
        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        if os.path.exists('%snet.pth'%self.appName[0]) == True:
            self.eval_net.load_state_dict(torch.load('%snet.pth'%self.appName[0]))
            self.target_net.load_state_dict(torch.load('%snet.pth'%self.appName[0]))
        
        self.memory_counter = 0 
        self.MEMORY_CAPACITY = MEMORY_CAPACITY                                                # for storing memory
        self.memory = {}                                                        # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                          
        self.learn_step_counter = 0


    def  choose_action(self,state):
        s = copy.deepcopy(state)
        tmp = np.random.uniform()
        if tmp < EPSILON:
            action,_,location,_,(hx,cx) = self.eval_net.forward(s)
            action = action[0][0]
            location = location[0][0]
            location = (int(location / 256), int(location % 256))
        else: 
            action = np.random.randint(0, self.action_space)
            #print("action")
            #print(action)
            _,(hx,cx) = s
            location = np.random.randint(0, 255*255)
            location = (int(location / 256), int(location % 256))
            #print("location")
            #print(location)
        if tmp > 0.9:
            action = 7
        if tmp>0.99:
            action = 8
        return action,location,(hx,cx)
    
    def store_transition(self, s, a, r, s_):
        s_1,(hx,cx) = s 
        s_img,action_possible = s_1
        s_1_,(hx_,cx_) = s 
        s_img_,action_possible_ = s_1

        transition = (s_img,action_possible,hx,cx,a,r,s_img_,action_possible_,hx_,cx_)
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = transition
        self.memory_counter += 1
    
    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        temp = MEMORY_CAPACITY if self.memory_counter > MEMORY_CAPACITY else self.memory_counter 

        sample_index = np.random.choice(temp, BATCH_SIZE)                       # 在[0, temp)内随机抽取16个数，可能会重复

        b_memory = dict([(i , self.memory[sample_index[i]]) for i in range(BATCH_SIZE)])     # 抽取16个索引对应的16个transition，存入b_memory

        b_img = torch.cat([b_memory[x][0] for x in b_memory],dim = 0)
        b_action_possible = torch.cat([b_memory[x][1] for x in b_memory],dim = 0)

        b_hx = torch.cat([b_memory[x][2] for x in b_memory],dim = 0)

        b_cx = torch.cat([b_memory[x][3] for x in b_memory],dim = 0)

        # 将32个s抽出并存储到b_s中，b_s为32行1列
        b_action =  torch.from_numpy(np.array([[b_memory[x][4][0]] for x in b_memory])).long()
        b_location =  torch.from_numpy(np.array([[b_memory[x][4][1]] for x in b_memory])).long()

        b_r = np.array([[b_memory[x][5]] for x in b_memory],dtype=int).astype(int)

        b_img_ = torch.cat([b_memory[x][6] for x in b_memory],dim = 0)

        b_action_possible_ = torch.cat([b_memory[x][7] for x in b_memory],dim = 0)

        b_hx_ = torch.cat([b_memory[x][8] for x in b_memory],dim = 0)

        b_cx_ = torch.cat([b_memory[x][9] for x in b_memory],dim = 0)


        b_1 = (b_img,b_action_possible)
        b_s = (b_1,(b_hx,b_cx))
        
        b_1_ = (b_img_,b_action_possible_)
        b_s_ = (b_1_,(b_hx_,b_cx_))


        # 获取16个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        _,q_eval_action_value, _,q_eval_location_value, (q_eval_hx , q_eval_cx) = self.eval_net(b_s)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next_action ,q_next_action_value, q_next_location,q_next_location_value , (q_next_hx , q_next_cx) = self.target_net(b_s_)
        q_next_action_value.detach()
        q_next_location_value.detach()
        q_next_hx.detach()
        q_next_cx.detach()
        #原本
        # q_eval_action = q_eval_action_value.cpu().gather(1,b_action)
        # q_eval_location = q_eval_location_value.cpu().gather(1,b_location)
        # q_target_action = torch.from_numpy(b_r).long() + GAMMA * q_next_action_value.cpu().gather(1,torch.from_numpy(q_next_action).long())
        # q_target_location = torch.from_numpy(b_r).long() + GAMMA * q_next_location_value.cpu().gather(1,torch.from_numpy(q_next_location).long())
        # loss_a = self.loss_func(q_eval_action, q_target_action)
        # loss_loc = self.loss_func(q_eval_location, q_target_location)
        # loss = loss_a + loss_loc

        q_eval_action = q_eval_action_value.cpu().gather(1,b_action)
        q_eval_location = q_eval_location_value.cpu().gather(1,b_location)
        q_target_action = q_next_action_value.cpu().gather(1,torch.from_numpy(q_next_action).long())
        q_target_location = q_next_location_value.cpu().gather(1,torch.from_numpy(q_next_location).long())
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_eval = torch.mul(q_eval_action,q_eval_location)
        q_target = torch.from_numpy(b_r).long() + GAMMA * torch.mul(q_target_action,q_target_location)
        loss = self.loss_func(q_eval,q_target)
        # 输入16个评估值和16个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                              # 清空上一步的残余更新参数值
        loss.backward()                                                         # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                                   # 更新评估网络的所有参数
