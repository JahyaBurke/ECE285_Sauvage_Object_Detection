# Create Dataloader Class
# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import xml.etree.ElementTree as ET
from PIL import Image

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
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        
        # Compose the path to the image file from the image_dir + image_name
        image_path = os.path.join(self.image_dir, self.image_filenames[ind])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_filenames[ind])
        
        # Load the image
        #image = read_image(image_path)
        image = Image.open(image_path)
        image = transform(image)
        
        anno = ET.parse(annotation_path)
            
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            try:
                bbox.append([
                    int(float(bndbox_anno.find(tag).text)) - 1
                    for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            except:
                print(annotation_path)
            name = obj.find('name').text.lower().strip()
            label.append(self.classes[name])
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Return the image and its label
        return (image, bbox, label)

def create_split_loaders(data_dir, batch_size, seed=0, 
                         p_val=0.1, p_test=0.2, shuffle=True, 
                         show_sample=False):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """
    
    # Get create a Dataset object
    dataset = PascalVOCDataloader(data_dir)

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    
    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if torch.cuda.is_available():
        num_workers = 1
        pin_memory = True
        
    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=sample_train, num_workers=num_workers, 
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size, 
                             sampler=sample_test, num_workers=num_workers, 
                              pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers, 
                              pin_memory=pin_memory)

    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)