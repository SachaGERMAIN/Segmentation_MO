import torch
import torch.nn as nn
from MO.dataset_test import testDataset,test
import segmentation_models_pytorch as smp
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm 
from time import sleep
import matplotlib as mpl
#from torch.utils.tensorboard import SummaryWriter

PATH = "Unet++_100_epochs_batchsize_12_with_val.pt"
                                                
IMAGE_PATH = './data/processed/deepLearning_training'
MASK_PATH = './data/processed/deepLearning_training_export_simpleSegmentation'
                                               
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_data=testDataset(img_path=IMAGE_PATH,mask_path=MASK_PATH)
val_loader=DataLoader(val_data,batch_size=1,shuffle=True)

criterion = nn.CrossEntropyLoss()

model=torch.load(PATH)
model.cuda()

model.eval()


with tqdm(val_loader) as tepoch:
    for images, masks, xray, mask_init in tepoch:
        tepoch.set_description(f"Epoch 1")      

        images,masks=images.to(device=device),masks.to(device=device)
        masks1=torch.squeeze(masks,1).long()

        # Forward pass

        outputs = model(images)
        loss = criterion(outputs, masks1)  
        tp, fp, fn, tn = smp.metrics.get_stats(outputs.argmax(dim=1,keepdim=True), masks, mode='multilabel', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")  
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")     

        with torch.no_grad():
            colormaps=ListedColormap(['darkorange','gold','lawngreen']).resampled(256)
            test(colormaps,outputs,masks1,xray,mask_init)
            
        tepoch.set_postfix(loss=loss.item(), iou=iou_score.item(), f1_score=f1_score.item())
        sleep(0.1)
