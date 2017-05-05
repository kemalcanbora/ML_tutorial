import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
sequence_length = 28   # Time step
input_size = 28
hidden_size = 128      # number of hidden unit
num_layers = 2         # number of rnn layer
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.05

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# 繼承 nn.Module
# RNN model (Many to one)
class RNN(nn.Module):                  
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)  
        # input & output will has batch size as 1s dimension 
        self.outlayer = nn.Linear(hidden_size,num_classes)        # Linear = Dense layer?                                                                
    
    def forward(self,x):
        # Forward propagate RNN
        out, _ = self.rnn(x, None)  
        
        # Decode hidden state of last time step
        out = self.outlayer(out[:, -1, :])  
        return out
     
    
    
rnn = RNN(input_size, hidden_size, num_layers, num_classes)
print(rnn)

# Loss and Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, sequence_length, input_size))     # reshape x to (batch, time_step, input_size)
        labels = Variable(labels)        # batch y
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, sequence_length, input_size))
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total)) 
