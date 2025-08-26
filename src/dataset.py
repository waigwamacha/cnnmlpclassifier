
import os
import torch

import nibabel as nib

from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, df, id_col, agebracket, target_col, root_dir, transform=None): #csv_file,
        
        self.annotations = df #pd.read_csv(csv_file).sample(5000) # We will just use random 1000 images from the training images
        self.id = id_col
        self.agebracket = agebracket
        self.target = target_col
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        img_name = self.annotations.iloc[index][self.id]
        agebracket = self.annotations.iloc[index][self.agebracket]
        img_path = os.path.join(self.root_dir, img_name)
        label = self.annotations.iloc[index][self.target]

        nifti_image = nib.load(img_path) 
        image = nifti_image.get_fdata() 
        
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        #print(f"Unsqueezed: {image_tensor.shape}")

        image_tensor = torch.mean(image_tensor, -1)

        agebracket_tensor = torch.tensor([agebracket], dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0) #torch.tensor(int(label))
        #print(f"Expand dims: {image_tensor.shape}")

        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, agebracket_tensor, label_tensor
    

