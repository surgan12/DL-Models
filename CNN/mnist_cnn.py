import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import torch.optim as optim
#training set
use_cuda = torch.cuda.is_available()

classes=['0','1','2','3','4','5','6','7','8','9']
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1,))])

root='./data'
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)


batch_size=500

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
shuffle=False)
dataiter = iter(train_loader)

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.batch_1=nn.BatchNorm2d(20)
        self.conv2=nn.Conv2d(20, 50, 5, 1)
        self.conv3=nn.Conv2d(50,60,5,1)
        self.fc1 = nn.Linear(2*2*60, 500)
        self.batch_2=nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.batch_1(x)
        x = F.sigmoid(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*60)
        x = F.sigmoid(self.fc1(x))
        x = self.batch_2(x)
        x = self.fc2(x)
        return x

model=CNN_MNIST()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

device = torch.device("cuda:0")
if use_cuda:
    model.cuda()
else:
    print("cuda not found")
criteria=nn.CrossEntropyLoss()
for epoch in range(0,10):
    ave_loss=0.0
    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        if use_cuda:
            inputs,labels=inputs.cuda(),labels.cuda()
        optimizer.zero_grad()
        output=model(inputs)
        loss=criteria(output,labels)
        loss.backward()
        optimizer.step()
        ave_loss+=loss.item()
        if i%100==99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, ave_loss / 100))
            ave_loss = 0.0
correct=0
total=0
with torch.no_grad():
    for data in test_loader:
        images,labels=data
        if use_cuda:
            images , labels = images.cuda() , labels.cuda()
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print('Accuracy of the network on the  test images: %0.6f %%' % (
    100 * correct / total))
