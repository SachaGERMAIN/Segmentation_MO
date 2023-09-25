import numpy as np 
import tifffile as tiff
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from glob import glob
import os

IMAGE_PATH = './data/sol_seul/image'
MASK_PATH = './data/sol_seul/mask'


class NormalisationDataset(Dataset):

    """
    class to store imgp and acces images
    indexation return image loaded as Tensor
    """
    def __init__(self, img_directory, patch_size=(256, 256)):
        self.images = glob(f"{img_directory}/*.tiff")  
        self.transform=T.ToTensor()

    def __getitem__(self, idx: int) -> Tensor:
        imgp=self.images[idx]
        img = tiff.imread(imgp)
        img=img.astype('int32')
        img=self.transform(img)
        img=img/65536
        #permutation
        return img


    def __len__(self):
        return len(self.images)



def get_normalize_params(data_dir: str) -> tuple[Tensor, Tensor]:
    """
    take normalisation dataset and output mean and std over 3 channels for
    large dataset (using batch processing).
    mean and std are tensor like
    """
    dataset = NormalisationDataset(img_directory=data_dir)
    dataloader = DataLoader(dataset, batch_size=16)  # store data in datloader
    channels_sum, channels_squared_sum, num_batches = (
        0,
        0,
        0
    )  # initialize channel stats

    for data in tqdm(dataloader,unit='batch'):
        # Mean over batch, height and width but no channels
        data_s=data**2
        channels_sum += torch.mean(data.float(),dim=[0,2,3])
        channels_squared_sum += torch.mean(data_s.float(),dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches  # compute mean over batches
    std = (
        channels_squared_sum / num_batches - mean**2
    ) ** 0.5  # compute std over batches

    return mean, std 

class testDataset(Dataset):
    
    def __init__(self, img_path, mask_path):
        self.images = glob(f"{img_path}/*.tiff")  
        self.masks = glob(f"{mask_path}/*.tiff")
        self.transform=T.ToTensor()
        self.img_path=img_path

    def __len__(self):
        return len(self.images)
       
    def __getitem__(self, idx: int)-> Tensor:
        imgp=self.images[idx]
        maskp=self.masks[idx]
        #Ouverture en np.array
        image=tiff.imread(imgp)
        mask=tiff.imread(maskp)
        #changer les val de masque
        mask[mask == 1] = 0
        mask[mask == 2] = 1
        mask[mask == 3] = 2       
        #passage en Tensor
        image=image.astype('int32')
        mask=mask.astype('int32')
        imaget=self.transform(image)
        maskt=self.transform(mask)
        #rescale entre 0 et 1
        imaget=imaget/65536
        #normalize
        meani,stdi=get_normalize_params(self.img_path)
        norm_im=T.Normalize(mean=meani,std=stdi)
        image_n=norm_im(imaget)

        return image_n, maskt, image, mask
    
class inf_Dataset(Dataset):
    
    def __init__(self, img_path):
        self.images = glob(f"{img_path}/*.tiff")  
        self.transform=T.ToTensor()
        self.img_path=img_path

    def __len__(self):
        return len(self.images)
       
    def __getitem__(self, idx: int)-> Tensor:
        imgp=self.images[idx]
        #Ouverture en np.array
        image=tiff.imread(imgp)     
        #passage en Tensor
        image=image.astype('int32')
        imaget=self.transform(image)
        #rescale entre 0 et 1
        imaget=imaget/65536
        #normalize
        meani,stdi=get_normalize_params(self.img_path)
        norm_im=T.Normalize(mean=meani,std=stdi)
        image_n=norm_im(imaget)

        return image_n, image
    


# train_loader=DataLoader(train_data,batch_size=1,shuffle=True)

# https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#sphx-glr-tutorials-colors-colormap-manipulation-py




def test(colormaps,image,mask,xray,mask_init,Nr=2,Nc=2):
    fig,axs=plt.subplots(Nr,Nc)
    image,mask=image.cpu(),mask.cpu()
    image,mask=image.argmax(dim=1),torch.squeeze(mask,0)
    imagen,maskn=image.numpy()[0],mask.numpy()
    data=[imagen,maskn]       
    axs[0,0].pcolormesh(data[0], cmap=colormaps, rasterized=True)
    axs[0,0].set_title('Prediction')
    axs[0,1].pcolormesh(data[0], cmap=colormaps, rasterized=True)
    axs[0,1].set_title('Colorized Mask')
    axs[1,0].imshow(xray.squeeze(dim=0),cmap='gray',origin='lower')
    axs[1,0].set_title('Original image')
    axs[1,1].imshow(mask_init.squeeze(dim=0),cmap='gray',origin='lower')
    axs[1,1].set_title('Original mask')
    fig.set_size_inches(10,10)
    plt.show()


