import torch
from torchvision import transforms
from PIL import Image

from model import Yolov1
from bboxes_and_plotting import cellboxes_to_boxes, plot_image
from algorithms import non_max_supression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
model.load_state_dict(torch.load("model_epoch50.pth", map_location=torch.device('cpu')))
model.eval()
model = model.to(device)

# Test for each image
image_path = "data/Images/000002.jpg"
image = Image.open(image_path)

# Transform the image
transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
image = transform(image).unsqueeze(0)
image = image.to(device)

# Forward pass
with torch.no_grad():
    output = model(image)

# Get bboxes by predicting bboxes for given image
bboxes = cellboxes_to_boxes(output)

# Remove multiple bboxes for same object
bboxes = non_max_supression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

# Plot the image with bboxes
plot_image(image[0].permute(1,2,0).to('cpu'), bboxes)
