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
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
import matplotlib.pylab as plt
import sys
from torch.utils.tensorboard import SummaryWriter

#device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#tensorboard config
writer = SummaryWriter("runs/MNIST2")

#hyper-parameters
input_size = 784 # 28 * 28 pixles 
hidden_size = 100 # numnber of params in hidden layers
num_classes = 10 # num digits to classify
num_epochs = 2
batch_size = 100
learning_rate = 0.01

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

image_grid = torchvision.utils.make_grid(samples)
writer.add_image('Mnist Images', image_grid)
#writer.close()

#sys.exit()

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

writer.add_graph(model, samples.reshape(-1, 28*28))


#training
n_total_steps = len(train_loader)
running_loss = 0
running_correct =0
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

        running_loss += loss.item()
        _, predictions = torch.max(preds,1)
        running_correct += (predictions==labels).sum().item()
        if (i+1)% 100 == 0:
            print(f'epoch: {epoch} / {num_epochs}, steps: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')
            writer.add_scalar('training loss',running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('training accuracy',running_correct / 100, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0


#evaluation pahse 
tensorboard_labels = []
tensorboard_preds = []
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

        # for tesnsorboard
        class_preds = [F.softmax(output, dim=0) for output in outputs]
        tensorboard_preds.append(class_preds)
        tensorboard_labels.append(labels)

    tensorboard_preds = torch.cat([torch.stack(batch) for batch in tensorboard_preds]) # 10k * 10
    tensorboard_labels = torch.cat(tensorboard_labels) # 10k * 1

accu = n_corrects / n_samples * 100
print (f' test accuray is: {accu}')

classes = range(10)
for i in classes:
    labels_i = tensorboard_labels==i 
    preds_i = tensorboard_preds[:,i]
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()

    