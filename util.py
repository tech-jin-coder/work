import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import sys
import copy
class Util(object):
    def __init__(self,transform,device,width,height):
        self.action_possible = torch.zeros((1,7))
        self.pre_action_possible = torch.zeros((1,7))
        self.color = {15:np.array([255,255,255]),14:np.array([17*14,17*14,17*14]),13:np.array([17*13,17*13,17*13]),12:np.array([17*12,17*12,17*12]),11:np.array([17*11,17*11,17*1]),10:np.array([17*10,17*10,17*10]),
            9:np.array([17*9,17*9,17*9]),8:np.array([17*8,17*8,17*8]),7:np.array([17*7,17*7,17*7]),6:np.array([17*6,17*6,17*6]),5:np.array([17*5,17*5,17*5]),
            4:np.array([17*4,17*4,17*4]),3:np.array([17*3,17*3,17*3]),2:np.array([17*2,17*2,17*2]),1:np.array([17,17,17]),0:np.array([0,0,0])}
        self.img = np.full((1,256,256),255).astype(int)
        self.img1 = np.full((1,256,256),255).astype(int)
        self.chose = np.zeros((256,256)).astype(int)
        self.chose1 = np.zeros((256,256)).astype(int)
        self.transform = transform
        self.device = device
        self.width = width
        self.height = height
        self.pre_locator_list = []

    def getpos(self,bounds):
        flag = 0
        tmp = ""
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        for i in bounds:
            if i == '[' and flag ==0:
                flag = 1
                continue
            if i == ']' and flag ==1:
                x1 = int(tmp)
                tmp = ""
                continue
            if i == '[' and flag ==1:
                flag = 2
                continue
            if i == ']' and flag ==2:
                x2 = int(tmp)
                tmp = ""
                continue
            if i == ',' and flag ==1:
                y1 = int(tmp)
                tmp = ""
                continue
            if i == ',' and flag ==2:
                y2 = int(tmp)
                tmp = ""
                continue
            tmp += i
        return x1,y1,x2,y2

    def make_pic(self,locator_list):
        self.img = copy.deepcopy(self.img1)
        self.chose = copy.deepcopy(self.chose1)

        for loc in locator_list:
            cnt = loc[2]
            bounds = loc[1]['bounds']
            y1,x1,y2,x2 = self.getpos(bounds)
            #print(x1,x2,y1,y2)
            # x [0,width], y [0,height]
            x1 = (int)((float)(x1)/self.width * 256)
            x2 = (int)((float)(x2)/self.width * 256) if (int)((float)(x2)/self.width * 256) > x1 else x1 +1
            y1 = (int)((float)(y1)/self.height * 256)
            y2 = (int)((float)(y2)/self.height * 256) if (int)((float)(y2)/self.width * 256) > y1 else y1 +1
            #print(x1,x2,y1,y2)
            x1 = x1 + 1
            x2 = x2 - 3 if (x2 - 3) > x1 else x1 + 1
            y1 = y1 + 1
            y2 = y2 - 3 if (y2 - 3) > y1 else y1 + 1
            # 原本的方式
            # for i in range(x1,x2):
            #     for j in range(y1,y2):
            #         self.chose[i][j] = (self.chose[i][j] | cnt)
            #         self.img[0][i][j]= self.color[15-self.chose[i][j]][0]

            #用一个像素点代替
            self.chose[int((x1+x2)/2)][int((y1+y2)/2)] = (self.chose[int((x1+x2)/2)][int((y1+y2)/2)] | cnt)
            self.img[0][int((x1+x2)/2)][int((y1+y2)/2)]= self.color[15-self.chose[int((x1+x2)/2)][int((y1+y2)/2)]][0]
        return self.img


    def parse_page_sourse(self,source):
        self.action_possible = torch.zeros((1,7))
        locator_list = []
        tree = ET.fromstring(source)
        for node in tree.iter():
            checkable = node.get('checkable')
            clickable = node.get('clickable')
            long_clickable = node.get('long-clickable')
            scrollable = node.get('scrollable')
            focusable = node.get('focusable')
            cnt = 0
            if(focusable == "true" and (checkable == "true" or clickable == "true" or long_clickable == "true" or scrollable == "true") ):
                tt = []
                if(checkable == "true"):
                    cnt = cnt | 1
                    tt.append(2)
                if(clickable == "true"):
                    cnt = cnt | (1<<1)
                    tt.append(0)
                if(long_clickable == "true"):
                    cnt = cnt | (1<<2)
                    tt.append(1)
                if(scrollable == "true"): 
                    cnt = cnt | (1<<3)
                    tt.append(3)
                    tt.append(4)
                    tt.append(5)
                    tt.append(6)
                if [node.tag,node.attrib,cnt] not in self.pre_locator_list:
                    locator_list.append([node.tag,node.attrib,cnt])
                    for i in tt:
                        self.action_possible[0][i] = 1
        return locator_list,self.action_possible
        
    def get_state_img(self,page):
        locator_list,action_possible1 = self.parse_page_sourse(page)
        # print(locator_list)
        if locator_list:
            self.pre_locator_list = copy.deepcopy(locator_list)
            self.pre_action_possible = copy.deepcopy(action_possible1)
        else:
            locator_list = copy.deepcopy(self.pre_locator_list)
            action_possible1 = copy.deepcopy(self.pre_action_possible)
        img = self.make_pic(locator_list)

        #转图片显示
        # tmp = np.zeros((3,256,256),dtype=int)
        # tmp[0,:,:] = img
        # tmp[1,:,:] = img
        # tmp[2,:,:] = img
        # tmp = np.moveaxis(tmp,0,-1)
        # img_t = Image.fromarray(tmp.astype(np.uint8), 'RGB')
        # img_t.save('my1.png')
        # img_t.show()

        img = torch.tensor(np.expand_dims(img,axis=0))
        #(1,1,256,256)
        img = img.to(self.device)
        action_possible1 = action_possible1.to(self.device)
        return (img,action_possible1)
