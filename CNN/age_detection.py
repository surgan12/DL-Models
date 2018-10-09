import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import io, transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import math
import torch.nn.functional as F


classes=['MIDDLE','YOUNG','OLD']
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])

class AgeDataSet(Dataset):
	def __init__(self,csv,root,transform=None):
		self.root=root
		self.data=pd.read_csv(csv)
		self.X=np.array(self.data.iloc[:,0]).reshape(-1,1)
		y=self.data.iloc[:,1].apply(classes.index)
		self.Y=np.array(y).reshape(-1,1)
		
		self.transform=transform
		
	def __len__(self):
		return len(self.X)
	def __getitem__(self,idx):
		path=self.root
		item=self.X[idx][0]
		item=path+item
		image = io.imread(item)
		label=self.Y[idx][0]
		#print(label)
		if self.transform:
			image = self.transform(image)
		
		
		return {'image':image,'label':label}	

age=AgeDataSet('/home/surgan/Desktop/DL-Models/CNN/Train_Age/Train/train.csv','/home/surgan/Desktop/DL-Models/CNN/Train_Age/Train/',transform=transform)

train_size=int(0.8*len(age))
test_size=len(age)-train_size
train_dataset, test_dataset = torch.utils.data.random_split(age, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=15,
                        shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=15	,
                        shuffle=False, num_workers=4)


dataiter = iter(train_loader)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.batch_1=nn.BatchNorm2d(20)
        self.conv2=nn.Conv2d(20, 50, 5, 1)
        self.conv3=nn.Conv2d(50,60,5,1)
        self.fc1 = nn.Linear(2*2*60, 500)
        self.batch_2=nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.batch_1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*60)
        x = F.relu(self.fc1(x))
        x = self.batch_2(x)
        x = self.fc2(x)
        return x

model=CNN()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

criteria=nn.CrossEntropyLoss()
for epoch in range(25):
    ave_loss=0.0
    for i,data in enumerate(train_loader,0):
        inputs=data['image']
        labels=data['label']
        
        inputs=torch.from_numpy(np.array(inputs))
        labels=torch.from_numpy(np.array(labels))
        optimizer.zero_grad()
        output=model(inputs)
        
        
        loss=criteria(output,labels)
        loss.backward()
        optimizer.step()
        ave_loss+=loss.item()
        if i%200==199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, ave_loss / 100))
            ave_loss = 0.0
correct=0
total=0
with torch.no_grad():
    for data in test_loader:
        inputs=data['image']
        labels=data['label']
        
        inputs=torch.from_numpy(np.array(inputs))
        labels=torch.from_numpy(np.array(labels))
        outputs=model(inputs)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print('Accuracy of the network on the  test images: %0.6f %%' % (
    100 * correct / total))
