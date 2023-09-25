import torch
import torch.nn as nn
from MO.dataset_test import testDataset,test
import segmentation_models_pytorch as smp
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm 
from time import sleep
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.tensorboard import SummaryWriter

PATH = "Unet++_100_epochs_batchsize_5.pt"
                                                
IMAGE_PATH = './data/deepLearning_training'
MASK_PATH = './data/deepLearning_training_export_simpleSegmentation'

def plot3(colormaps, sample,Nr=3,Nc=4):

    fig,axs=plt.subplots(Nr,Nc)
    label_list=['Prediction','Colorized Mask','Original Image','Original Mask']
    sample=sample.cpu()

    for i in range(len(sample[0])):

        sample[0][i]=sample[0][i].argmax(dim=1).numpy()[0]
        sample[1][i]=torch.squeeze(sample[1][i],0).numpy()

    for x in range(Nc):
        for y in range(Nr):
            if x<2:
                axs[x,y].pcolormesh(sample[x][y],cmap=colormaps,rasterized=True)   
            if x<2:
                axs[x,y].imshow(sample[x][y].squeeze(dim=0),cmap='gray',origin='lower')
            axs[x,y].set_title(label_list[x])
    fig.set_size_inches(6,6)
    plt.show()

                                              
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_data=testDataset(img_path=IMAGE_PATH,mask_path=MASK_PATH)
val_loader=DataLoader(val_data,batch_size=3,shuffle=True)


sample=next(iter(val_loader))
sample=torch.cat(sample).cuda() # pb dimension, séparer images et masks en deux listes
criterion = nn.CrossEntropyLoss()

model=torch.load(PATH)
model.cuda()

model.eval()

n_total_steps = len(val_loader)


for i in range(len(sample[0])):

    sample[0][i]=sample[0][i].to(device=device)
    sample[1][i]=torch.squeeze(sample[1][i],1).long().to(device=device)

    # Forward pass

    sample[0][i] = model(sample[0][i])
    loss = criterion(sample[0][i], sample[1][i])  
    tp, fp, fn, tn = smp.metrics.get_stats(sample[0][i].argmax(dim=1,keepdim=True), sample[1][i], mode='multilabel', threshold=0.5)
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")  
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro") 
    print(f"Image {i+1}/{len(sample[0])}")    
    print(f"Loss = {loss.item()}, IoU score = {iou_score.item()}, F1 score = {f1_score.item()}")

with torch.no_grad():
    colormaps=ListedColormap(['darkorange','gold','lawngreen']).resampled(256)
    plot3(colormaps,sample)
        



