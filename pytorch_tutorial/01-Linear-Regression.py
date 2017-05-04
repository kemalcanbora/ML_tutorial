import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Hyper Parameters
input_size = 1
hidden_size = 10
output_size = 1
num_epochs = 600
learning_rate = 0.01

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        out = F.relu(self.hidden(x))      # activation function for hidden layer
        out = self.predict(out)             # linear output
        return out



net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

for epoch in range(num_epochs):

    # Forward + Backward + Optimize
    optimizer.zero_grad()  
    prediction = net(x)
    loss = loss_func(prediction, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [%d/%d], Loss: %.4f' 
               %(epoch+1, num_epochs, loss.data[0]))

# Plot the graph
predicted = net(x).data.numpy()
plt.plot(x.data.numpy(), y.data.numpy(), 'ro', label='Original data')
plt.plot(x.data.numpy(), predicted, label='Fitted line', lw=5)
plt.legend()
plt.show()




