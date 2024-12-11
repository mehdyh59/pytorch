""" Compare RNN, GRU and LSTM using MNIST dataset
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

#Models to evaluate
models =['gru','rnn','lstm']

#device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#tensorboard config
writer_dict={}
for model in models:
    writer_dict[model]= SummaryWriter("runs/MNIST-RNN-"+model)


#hyper-parameters
#input_size = 784 # 28 * 28 pixles 
input_size = 28 # in order to work with RNNs, we cosider input size 784 as 28 senquences each of size 28
sequence_length = 28
num_layers = 2 # stacking two rnns to improve performance
hidden_size = 128 # numnber of params in hidden layers
num_classes = 10 # num digits to classify
num_epochs = 2
batch_size = 100
learning_rate = 0.001


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))]) 

#MNIST data prepration
training_data = torchvision.datasets.MNIST(root='./data/MNIST', train=True,transform=transform, download=True)
test_data = torchvision.datasets.MNIST(root='./data/MNIST/', train=False, transform=transform,download=False)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False)

""" batch = iter(train_loader)
samples, labels = next(batch) """

""" print(samples.shape, labels.shape)

for i in range(100):
    plt.subplot(10,10, i+1)
    plt.imshow(samples[i][0],cmap='grey')
plt.show() """

#image_grid = torchvision.utils.make_grid(samples)
#writer.add_image('Mnist Images', image_grid)
#writer.close()
#sys.exit()

# modeling part
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, model_type='rnn') -> None:
        super(RNN, self).__init__()
        self.rnn, self.gru, self.lstm = None, None, None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        match model_type:
            case 'rnn':
                self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            case 'gru':
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            case 'lstm':
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # batch_first = True --> x = (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        # initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # (num_layers, batch size, hidden size)
        out = None
        match self.model_type:
            case 'rnn':
                out, _ = self.rnn(x, h0)
            case 'gru':
                out, _ = self.gru(x, h0)            
            case 'lstm':
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # (num_layers, batch size, hidden size)
                out, _ = self.lstm(x, (h0,c0))

        #the out shape is: batch_size, seq_length, hidden_size
        # therefore out (N, 28, 128)
        # we only need the last sequence; out -> (N, 128)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

models_dict = {}
for model in models:
    models_dict[model] = RNN(input_size, hidden_size, num_classes, num_layers,model_type=model)


# loss
criterion = nn.CrossEntropyLoss()

#optimizers
optimizer_dict={}
for model in models:
    optimizer_dict[model] = torch.optim.Adam(models_dict[model].parameters(),lr=learning_rate)

#writer.add_graph(model, samples.reshape(-1, sequence_length, input_size))
#writer.close()
#sys.exit()

#training
def train (model, optimizer, writer, model_name) -> None:
    n_total_steps = len(train_loader)
    running_loss = 0
    running_correct =0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
        # original shape: [100, 1, 28,28] --> (batch_size, num_channels, pixels, pixels)
        # resized: [100, 28, 28]
            images = images.reshape(-1, sequence_length, input_size).to(device)
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
                print(f'model: {model_name}, epoch: {epoch} / {num_epochs}, steps: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')
                writer.add_scalar('training loss',running_loss / 100, epoch * n_total_steps + i)
                writer.add_scalar('training accuracy',running_correct / 100, epoch * n_total_steps + i)
                running_correct = 0
                running_loss = 0


for model in models:
    train(models_dict[model],optimizer_dict[model],writer_dict[model],model)



#evaluation pahse 
""" tensorboard_labels = []
tensorboard_preds = []
with torch.no_grad():
    n_corrects = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
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
print (f' test accuray is: {accu}') """

#classes = range(10)
#for i in classes:
#    labels_i = tensorboard_labels==i 
#    preds_i = tensorboard_preds[:,i]
 #   writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
 #   writer.close()


#save the model to be used in app
#torch.save(model.state_dict(),"mnist_ffn.pth")