import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import torch.optim as optim

use_cuda=torch.cuda.is_available()
print(use_cuda)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=15,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=15,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class CIFARNet(nn.Module):
	"""docstring for """
	def __init__(self):
		super(CIFARNet, self).__init__()
		self.conv1=nn.Conv2d(3,6,3,1,(1,1))

		self.max_1=nn.MaxPool2d(2,2)
		self.batch_2=nn.BatchNorm2d(6)
		self.conv2=nn.Conv2d(6,16,5)
		self.conv3=nn.Conv2d(16,32,3)
		self.batch_3=nn.BatchNorm2d(16)
		self.fc1=nn.Linear(2*2*32,120)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)
		
	def forward(self,x):
		x=self.max_1(F.relu(self.conv1(x)))
		x=self.batch_2(x)
		x=self.max_1(F.relu(self.conv2(x)))
		#x=self.batch_3(x)
		x=self.max_1(F.relu(self.conv3(x)))
		x=x.view(-1,2*2*32)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x
net=CIFARNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
if use_cuda:
	net.cuda()
else:
	print("not found")
for epoch in range(40):
	running_loss=0.0
	for i,data in enumerate(trainloader,0):

		inputs,labels=data
		if use_cuda:
			inputs,labels=inputs.cuda(),labels.cuda()
		optimizer.zero_grad()
		outputs=net(inputs)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i%2000==1:
			print('[%d, %5d] loss: %.3f'%(epoch + 1, i + 1, running_loss / 2000))
        	
print("finished training")
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if use_cuda:
            images , labels = images.cuda() , labels.cuda()
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))




