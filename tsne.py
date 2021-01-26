import numpy as np
from sklearn.manifold import TSNE
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
from torchvision import models

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



def main():
    TRAIN_DATASET = STL10(root="data", split="train", download=False, transform=ToTensor())
    tsne = TSNE(n_components=2)
    features = np.zeros([5000,512])
    labels = []
    model = models.resnet18(pretrained=False)
    model.fc = Identity()
    model.eval()
    print("Transforming images to features...")
    with torch.no_grad():
        percent = 0
        for xi, x in enumerate(TRAIN_DATASET):
            if xi % (len(TRAIN_DATASET)/100) == 0:
                print("{}%".format(percent))
                percent += 1
            c, y = x
            cd = model(torch.unsqueeze(c, 0)).numpy()
            features[xi] = cd[0]
            labels.append(y)

    res = tsne.fit_transform(features)
    tsneDF = pd.DataFrame(data = res, columns=['tsne dim 1', 'tsne dim 2'])
    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(30)
    ts = fig.add_subplot(1, 1, 1)
    ts.set_xlabel('Dimension 1', fontsize = 15)
    ts.set_ylabel('Dimension 2', fontsize = 15)
    ts.set_title('TSNE', fontsize = 20)

    ts.scatter(tsneDF['tsne dim 1'],
            tsneDF['tsne dim 2'], 
            c = cs_colors
            )
    recs=[]
    for i in range(len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    ts.legend(handles=recs, labels=cell_ids, title="cell")
    ts.grid()

if __name__ == '__main__':
    main()