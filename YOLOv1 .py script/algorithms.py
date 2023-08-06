def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
  """
  Calculates intersection over union

  Parameters:
    boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
    boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    box_format (str): midpoint/ corners, if boxes (x,y,w,h) or (x1, y1, x2, y2)

  Returns:
    tensor: Intersection over union for all examples

  Note:
    The `...` is used for indexing all elements in the preceding dimensions
    and `0:1` represents the range of indices to extract along the last dimension
  """

  if box_format == 'midpoint':
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

  elif box_format == 'corners':
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4] # Slicing this way to maintain the shape i.e (N,1) where, N is the number of bboxes

    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

  x1 = torch.max(box1_x1, box2_x1)
  y1 = torch.max(box1_y1, box2_y1)
  x2 = torch.max(box1_x2, box2_x2)
  y2 = torch.max(box1_y2, box2_y2)

  # .clamp(0) is for the case when they do not intersect
  intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0) # length * breadth

  box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1)) # Absolute so that area is not negative
  box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1)) # Absolute so that area is not negative

  # IOU = Area of intersection / Area of Union
  return intersection / (box1_area + box2_area - intersection + 1e-6)



"""# Non Max Supression Algorithm
- Step 1: Start with discarding all bounding boxes < probability threshold
- Step 2: Select the Bounding Box with the largest probability / score
- Step 3: Remove all the other bounding boxes with IoU > threshold

**Note**: Do for each class
"""
def non_max_supression(bboxes, iou_threshold, threshold, box_format="corners"):
  """
  Note: bboxes input should be list of bounding boxes
  i.e bboxes = [[1, 0.9, x1, y1, x2, y2], ..] # Each bounding box --> [class, probability, x1, y1, x2, y2]

  """

  # Validate the input
  assert type(bboxes) == list

  # Discard all the bounding box < probability threshold
  bboxes = [box for box in bboxes if box[1] > threshold]

  # Sort the bboxes in descending order based on their probabilities
  bboxes = sorted(bboxes, key=lambda x:x[1], reverse=True)

  # Create empty list for bboxes to append after NMS
  bboxes_after_nms = []

  while bboxes:
    chosen_box = bboxes.pop(0) # Select and remove the bounding box with largest probability from bboxes list

    # New list comprehension for different class or same class having IoU less than threshold
    bboxes = [
        box
        for box in bboxes
        if box[0] != chosen_box[0] # Checks if the class label of box is different from the class label of chosen_box
        or intersection_over_union(
            torch.tensor(chosen_box[2:]),
            torch.tensor(box[2:]),
            box_format = box_format,
        ) < iou_threshold # Checks if IoU < iou_threshold
    ]

    bboxes_after_nms.append(chosen_box)

  return bboxes_after_nms



"""# Mean Average Precision

- Step 1:  Get all bounding box predictions on our test set
- Step 2: Sort by **descending confidence score**
- Step 3: Calculate the **Precision** and **Recall** as we go through all outputs
- Step 4: Plot the **Precision-Recall graph**
- Step 5: Calculate Area under **Precision-Recall graph** which is the **average precision** for one class.
- Step 6: Similarly calculate **average precision** for all the other class
- Step 7: Calculate **mAP** i.e sum of all the average pecisions of all the classes / number of classes

Finally, all this was calculated given **specific IoU** threshold of 0.5, we need to redo all computations for many IoUs, example: 0.5, 0.55, 0.6, ..., 0.95. Then **average this** and this will be our **final result**. This is what is meant by mAP@0.5:0.005:0.95 (mAP at 0.5 with a step size of 0.05 upto 0.95).
"""
import torch
from collections import Counter

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint', num_classes=20):
  """
  Note: pred_boxes input should be list of bounding boxes
  i.e pred_boxes = [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]

  Similarly for true_boxes
  """

  average_precisions = [] # To store average precisions of each class
  epsilon = 1e-6 # For numerical stability

  for c in range(num_classes):
    detections = []
    ground_truths = []

    for detection in pred_boxes:
      if detection[1] == c:
        detections.append(detection)

    for true_box in true_boxes:
      if true_box[1] == c:
        ground_truths.append(true_box)

    # Calculate the count of unique elements in the ground_truths list and stores the result in the amount_bboxes variable
    # For eg. img 0 has 3 bboxes, img 1 has 5 bboxes then, amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    for key, val in amount_bboxes.items():
      amount_bboxes[key] = torch.zeros(val)
    # amount_boxes = {0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0])}

    # Sort the bboxes in descending order based on their probabilities
    detections.sort(key=lambda x: x[2], reverse=True)

    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)

    # If none exists for this class then we can safely skip
    if total_true_bboxes == 0:
      continue

    for detection_idx, detection in enumerate(detections):
      # Only filter images having same index
      ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

      num_gts = len(ground_truth_img)
      best_iou = 0

      # For selection of bbox having highest iou
      for idx, gt in enumerate(ground_truth_img):
        iou = intersection_over_union(
            torch.tensor(detection[3:]),
            torch.tensor(gt[3:]),
            box_format = box_format
        )

        if iou > best_iou:
          best_iou = iou
          best_gt_idx = idx

      # Categorizing either TP or FP
      if best_iou > iou_threshold:
        # Check if we haven't covered this bounding box before | '0' means we haven't covered
        if amount_bboxes[detection[0]][best_gt_idx] == 0:
          TP[detection_idx] = 1
          amount_bboxes[detection[0]][best_gt_idx] == 1 # Update that now it's covered
        else:
          FP[detection_idx] = 1
      else:
        FP[detection_idx] = 1

    # [1,1,0,1,0] --> [1,2,2,3,3]
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)

    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

    # By adding these initial values of 1 to precisions and 0 to recalls,
    # the code ensures that the precision and recall values start with the appropriate initial points.
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))

    # Calculate the average precision by using the trapezoidal rule to compute the area under the precision-recall curve
    average_precisions.append(torch.trapz(precisions, recalls))

  # Return mAP
  return sum(average_precisions) / len(average_precisions)

