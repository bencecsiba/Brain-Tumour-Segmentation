import torch

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold=0.5):

    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    
    bin_out = torch.where(outputs > threshold, 1, 0).type(torch.int16)
    labels = labels.type(torch.int16)
    
    intersection = (bin_out & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (bin_out | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our division to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch    

def calculate_iou(outputs, targets):
    # Convert outputs and targets to binary masks
    outputs = (outputs > 0.5).float()
    targets = (targets > 0.5).float()

    intersection = (outputs * targets).sum(dim=(1, 2))  # Calculate intersection
    union = (outputs + targets).sum(dim=(1, 2)) - intersection  # Calculate union

    iou = (intersection / (union + 1e-8)).mean()  # Calculate IoU

    return iou.item()