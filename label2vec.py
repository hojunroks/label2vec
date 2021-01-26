from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from torchvision import models
import torch
from torch import nn

TRAIN_DATASET = STL10(root="data", split="train", download=False, transform=ToTensor())
TEST_DATASET = STL10(root="data", split="test", download=False, transform=ToTensor())

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


if __name__=='__main__':
    model = models.resnet18(pretrained=True)
    pre_file = get_file('IWHX_finetuned.ckpt')
    if pre_file is not None:
            cls = Classifier.load_from_checkpoint(pre_file, feature_extractor=model)
            model.load_state_dict(cls.model.state_dict())
    model.fc = Identity()
    model.eval()
    labels = []
    count = [0,0,0,0,0,0,0,0,0,0]
    with torch.no_grad():
        for i in range(10):
            t = torch.zeros([500,512])
            labels.append(t)

        for xi, x in enumerate(TRAIN_DATASET):
            c, y = x
            cd = model(torch.unsqueeze(c, 0))
            labels[y][count[y]] = cd[0]
            count[y] += 1

        print(labels[0].shape)
        print(labels[0])
        std = torch.std(labels[0], dim=0)
        print(std.shape)
        print(std)
        print(torch.mean(std))