import torch

def calculate_iou(gt, prediction):
    intersection = torch.logical_and(gt, prediction)
    union = torch.logical_or(gt, prediction)
    return torch.sum(intersection) / torch.sum(union)