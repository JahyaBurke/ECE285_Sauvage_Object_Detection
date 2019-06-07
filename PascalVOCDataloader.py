# Create Dataloader Class
# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET

# Other libraries for data manipulation and visualization
import os
import numpy as np 
from data.util import read_image

class PascalVOCDataloader(Dataset):
    """Custom Dataset class for the PASCAL VOC Image Dataset.
    """
    
    def __init__(self, data_dir):
        """
        Args:
        -----
        - data_dir: directory of dataset: /datasets/ee285f-public/PascalVOC2012/

        Attributes:
        -----------
        - image_dir: The absolute filepath to the dataset images 
        - image_filenames: List of file names for images
        - annotation_dir: The absolute filepath to the dataset annotations
        - annotation_filenames: List of file names for annotations
        - classes: A dictionary mapping each label name to an int between [0, 19]
        """
        
        self.image_dir = os.path.join(data_dir+'JPEGImages')
        self.image_filenames = os.listdir(self.image_dir)
        self.image_filenames.sort()
        self.annotation_dir = os.path.join(data_dir, 'Annotations')
        self.annotation_filenames = os.listdir(self.annotation_dir)
        self.annotation_filenames.sort()
        self.classes = {'aeroplane':0, 'bicycle':1,'bird':2,'boat':3,'bottle':4,'bus':5,
                        'car':6, 'cat':7,'chair':8,'cow':9,'diningtable':10,'dog':11,
                        'horse':12, 'motorbike':13,'person':14,'pottedplant':15,
                        'sheep':16, 'sofa':17,'train':18,'tvmonitor':19}

        
    def __len__(self):
        
        # Return the total number of data samples
        return len(self.image_filenames)


    def __getitem__(self, ind):
        """Returns the image, its bounding boxes and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, bboxes, label)
        """
        
        # Compose the path to the image file from the image_dir + image_name
        image_path = os.path.join(self.image_dir, self.image_filenames[ind])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_filenames[ind])
        
        # Load the image
        image = read_image(image_path)
        #image = torch.from_numpy(img)[None]

        anno = ET.parse(annotation_path)
            
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(self.classes[name])
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Return the image and its label
        return (image, bbox, label)