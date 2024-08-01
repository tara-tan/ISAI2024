#create a torch dataloader for the dataset
import torch    
from torch.utils.data.dataloader import DataLoader
import torchvision
import os
from PIL import Image


class OsteoTorchDataset(torch.utils.data.Dataset):
    def __init__(self, itemsPath:list, labels:list, transform=None):
        
        self.itemsPath = itemsPath
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.itemsPath)

    def __getitem__(self,idx)->tuple[Image.Image,int]:
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        image = Image.open(self.itemsPath[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return torchvision.transforms.functional.pil_to_tensor(image), self.labels[idx]
        

class DataWrapper:
    def __init__(self, pathList,labelList, batch_size=16, num_workers=0, shuffle=False):
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize([244,244])
                                            # ,torchvision.transforms.Grayscale()
                                            ])
        osteoDataset = OsteoTorchDataset(pathList,labelList,transform)
        
        train,val = torch.utils.data.random_split(osteoDataset,[0.7,0.3])#MAY BUG
        
        self.trainLoader = DataLoader(train, batch_size = batch_size,shuffle=shuffle,num_workers=num_workers)
        self.valLoader = DataLoader(train, batch_size = batch_size,shuffle=shuffle,num_workers=num_workers)