import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
import pytorch_lightning as pl 

#hyper-parameters
input_size = 784 # 28 * 28 pixles 
hidden_size = 500 # numnber of params in hidden layers
num_classes = 10 # num digits to classify
num_epochs = 2
batch_size = 100
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))]) 

class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(LitNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out=self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        #forward pass
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self) :
        return torch.optim.Adam(self.parameters(),lr=learning_rate)
    
    def train_dataloader(self):
        training_data = torchvision.datasets.MNIST(root='./data/MNIST', train=True,transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,num_workers=15, persistent_workers=True, shuffle=True)

        return train_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        #forward pass
        preds = self(images)
        loss = F.cross_entropy(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def val_dataloader(self):
        validation_data = torchvision.datasets.MNIST(root='./data/MNIST', train=False,transform=transform, download=False)
        val_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,num_workers=15, persistent_workers=True, shuffle=False)

        return val_loader
    

if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=num_epochs,fast_dev_run=False)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)