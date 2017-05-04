import torch 
import torch.nn as nn
import torchvision.datasets as dsets           # 数据库模块
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.01

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True,                          # this is training data
                            transform=transforms.ToTensor(),     # Converts a PIL.Image or numpy.ndarray to
                            download=True)                       # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())

# plot one example
print(train_dataset.train_data.size())     # (60000, 28, 28)
print(train_dataset.train_labels.size())   # (60000)
plt.imshow(train_dataset.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.train_labels[0])
plt.show()

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)              # 要不要打乱数据 (打乱比较好)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,          # input height
                out_channels=16,        # n_filters
                kernel_size=5,          # filter size
                stride=1,               # filter movement/step
                padding=2,              # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1            
            ),                          # output shape (16, 28, 28)
            nn.ReLU(),                  # activation
            nn.MaxPool2d(kernel_size=2) # choose max value in 2x2 area, output shape (16, 14, 14)       
        )
        self.conv2 = nn.Sequential(    # input shape (16, 14, 14)
            nn.Conv2d(16,32,5,1,2),     # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2)             # output shape (32, 7, 7)
        )
        self.fc = nn.Linear(32*7*7, 10)
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)       # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc(out)
        return output   


cnn = CNN()
print(cnn)



# Loss and Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))



# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))



