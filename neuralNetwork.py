import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#fully connected network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes): #784
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Convolutional Neural Network
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channel = 1, num_classes = 10):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride =(2,2)) #cuts size in half
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters

#NN
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 6

#CNN
in_channels = 1

#load data
data_set = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
data_loader = DataLoader(dataset = data_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle=False, num_workers=0)

#init network
model = ConvolutionalNeuralNetwork().to(device)

#loss - cost function
criterion = nn.CrossEntropyLoss()

#learning algorithm, like gradient descent
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#train
def train():
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(data_loader):
            #get data in cpu
            data = data.to(device=device)
            targets = targets.to(device=device)

            #gets correct shape
            #data = data.reshape(data.shape[0], -1) #for NN

            #forward
            scores = model(data)
            loss = criterion(scores, targets)

            #backward
            optimizer.zero_grad()
            loss.backward()

            #gradient descent or adam step
            optimizer.step()

#check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1) #for NN

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

# checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
# save_checkpoint(checkpoint)
load_checkpoint(torch.load("my_checkpoint.pth.tar"))
check_accuracy(data_loader, model)
check_accuracy(test_loader, model)
