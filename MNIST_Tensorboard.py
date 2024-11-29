""" A simple feedforward net to classify minist data set, tensorboard is then used for visulisation tasks 
# MNIST download
# Dataloader, Transformation
# Multilayers Neural Net, activation fucntion
# Loss, backward and optimization
# training loop (batch training)
# model evaluation
# GPU support
# Visulization
"""
import torch 
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
import matplotlib.pylab as plt
#device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyper-parameters
input_size = 784 # 28 * 28 pixles 
hidden_size = 100 # numnber of params in hidden layers
num_classes = 10 # num digits to classify
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#MNIST data prepration
training_data = torchvision.datasets.MNIST(root='./data/MNIST', train=True,transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./data/MNIST/', train=False, transform=transforms.ToTensor(),download=False)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False)

batch = iter(train_loader)
samples, labels = next(batch)

""" print(samples.shape, labels.shape)

for i in range(100):
    plt.subplot(10,10, i+1)
    plt.imshow(samples[i][0],cmap='grey')
plt.show() """

# modeling part
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out=self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#training
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward pass
        preds = model(images)
        loss = criterion(preds, labels)

        #backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1)% 100 == 0:
            print(f'epoch: {epoch} / {num_epochs}, steps: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')

#evaluation pahse 
with torch.no_grad():
    n_corrects = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_corrects += (preds == labels).sum().item()

accu = n_corrects / n_samples * 100
print (f' test accuray is: {accu}')

    