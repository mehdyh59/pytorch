import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

#hyper-parameters
input_size = 784 # 28 * 28 pixles 
hidden_size = 100 # numnber of params in hidden layers
num_classes = 10 # num digits to classify
PATH = "mnist_ffn.pth"

#load model
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

#load model weights
model.load_state_dict(torch.load(PATH))
model.eval()


#image to tensor
def transform_image(images_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)) ,
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))]) 


    image = Image.open(io.BytesIO(images_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    image = image_tensor.reshape(-1, 28*28)
    output = model(image)
    _, pred = torch.max(output,1)
    return pred