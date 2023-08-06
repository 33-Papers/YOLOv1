import torch
from algorithms import non_max_supression

"""# Plot image"""
img_labels = {0:'aeroplane',
              1:'bicycle',
              2:'bird',
              3:'boat',
              4:'bottle',
              5:'bus',
              6:'car',
              7:'cat',
              8:'chair',
              9:'cow',
              10:'diningtable',
              11:'dog',
              12:'horse',
              13:'motorbike',
              14:'person',
              15:'pottedplant',
              16:'sheep',
              17:'sofa',
              18:'train',
              19:'tvmonitor'}

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    # Convert input image to numpy array
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        cls = int(box[0])
        prob = box[1]
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"

        # Calculate the top corner of bounding box
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        # Create Rectangle patch
        rect = Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add class and probability text
        text = f"{img_labels[cls]}: {prob:.2f}"
        ax.text(
            upper_left_x * width,
            upper_left_y * height - 10,
            text,
            fontsize=10,
            color="r",
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 2},
        )

    plt.show()
    return fig


"""# Get Bounding Box for true and predicted"""
def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda" if torch.cuda.is_available() else "cpu"):

  """
  Input images --> get all true boxes and predicted boxes
  """

  all_pred_boxes = []
  all_true_boxes = []

  # make sure model is in eval before get bboxes
  model.eval()
  train_idx = 0 # For each image

  for batch_idx, (x, labels) in enumerate(loader):
      x = x.to(device)
      labels = labels.to(device)

      with torch.no_grad():
          predictions = model(x)

      batch_size = x.shape[0]
      true_bboxes = cellboxes_to_boxes(labels)
      bboxes = cellboxes_to_boxes(predictions)

      # For every image in each batch --> NMS
      for idx in range(batch_size):
          nms_boxes = non_max_supression(
              bboxes[idx],
              iou_threshold=iou_threshold,
              threshold=threshold,
              box_format=box_format,
          )

          for nms_box in nms_boxes:
              all_pred_boxes.append([train_idx] + nms_box)

          for box in true_bboxes[idx]:
              # many will get converted to 0 pred
              if box[1] > threshold:
                  all_true_boxes.append([train_idx] + box)

          train_idx += 1

  model.train()
  return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """

    # Reshape the prediction and select the best bounding box
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    # Converting wrt image ratio
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1) # (batch_size, 7, 7, 1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]

    converted_bboxes = torch.cat((x, y, w_y), dim=-1) # batch_size, 7, 7, 4
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1) # batch_size, 7, 7, 1
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    ) # batch_size, 7, 7, 1
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds # batch_size, 7, 7, 6

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    # For each image in the batch extract all values from corr. bbox_idx
    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes