"""# Creating a Dataset"""

import torch
import os
import pandas as pd
from PIL import Image
import torchvision

class VOCDataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.label_dir = label_dir
    self.transform = transform
    self.S = S
    self.B = B
    self.C = C

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    label_path = os.path.join(self.label_dir, self.annotations.iloc[index,1])
    boxes = []

    with open(label_path) as f:
      for label in f.readlines():
        # List comprehension that converts each component of the line from string format to either float or integer
        class_label, x, y, width, height = [
            float(x) if float(x) != int(float(x)) else int(x)
            for x in label.replace("\n","").split()
        ]
        # Append bboxes for that particular label
        boxes.append([class_label, x, y, width, height])

    img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
    image = Image.open(img_path)
    boxes = torch.tensor(boxes)

    if self.transform:
      image = self.transform(image)

    label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
    for box in boxes:
      class_label, x, y, width, height = box.tolist()
      class_label = int(class_label)

      # i=cell row and j=cell column --> get the cell in which midpoint lies
      i , j = int(self.S * y), int(self.S * x)
      # Then again scales down to 0-1
      x_cell, y_cell = self.S * x - j, self.S * y - i
      width_cell, height_cell = (
          width * self.S,
          height * self.S
      )

      # Now fill in the label_matrix
      if label_matrix[i,j,20] == 0: # 20th index specifies if there is object or not
        label_matrix[i,j,20] = 1 # This means that cell has object
        box_coordinates = torch.tensor(
            [x_cell, y_cell, width_cell, height_cell]
        )
        label_matrix[i,j,21:25] = box_coordinates
        label_matrix[i,j,class_label] = 1 # Specifying that particular class is present

    return image, label_matrix


"""# Load Pascal VOC dataset"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'

transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

train_dataset = VOCDataset(
    "data/8examples.csv",
    transform=transform,
    img_dir = IMG_DIR,
    label_dir = LABEL_DIR,
)

test_dataset = VOCDataset(
    "data/8examples.csv",
    transform=transform,
    img_dir = IMG_DIR,
    label_dir= LABEL_DIR,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)