import numpy as np
import torch
import torch.nn as nn
import math


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class MetricCalculator(nn.Module):
    def __init__(self, n_classes):
        super(MetricCalculator, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """
        keep same as DiceLoss, convert target to one-hot format
        """
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _calculate_metrics(self, pred, target):
        """
        calculate precision, recall, f1 score and iou for one class
        """
        smooth = 1e-7
        pred = pred.float()
        target = target.float()

        # Intersection and Union
        intersect = torch.sum(pred * target)  # True Positive
        pred_sum = torch.sum(pred)  # Predicted Positive
        target_sum = torch.sum(target)  # Ground Truth Positive
        union = pred_sum + target_sum - intersect

        # Precision, Recall, IoU
        precision = (intersect + smooth) / (pred_sum + smooth)
        recall = (intersect + smooth) / (target_sum + smooth)
        iou = (intersect + smooth) / (union + smooth)

        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall + smooth)

        return precision, recall, f1_score, iou

    def forward(self, inputs, target, softmax=False):
        """
        calculate metrics for every class and average them
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        class_metrics = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "iou_score": []
        }
        total_correct = 0
        total_pixels = torch.numel(target)

        for i in range(self.n_classes):
            # get predictions and targets for current class
            pred = inputs[:, i]
            tgt = target[:, i]

            # calculate metrics for current class
            precision, recall, f1_score, iou = self._calculate_metrics(pred, tgt)
            class_metrics["precision"].append(precision.item())
            class_metrics["recall"].append(recall.item())
            class_metrics["f1_score"].append(f1_score.item())
            class_metrics["iou_score"].append(iou.item())

            #  (True Positive + True Negative)
            total_correct += torch.sum((pred > 0.5) == (tgt > 0.5)).item()

        # average metrics over all classes
        metrics = {
            "accuracy": total_correct / total_pixels,
            "precision": sum(class_metrics["precision"]) / self.n_classes,
            "recall": sum(class_metrics["recall"]) / self.n_classes,
            "f1_score": sum(class_metrics["f1_score"]) / self.n_classes,
            "iou_score": sum(class_metrics["iou_score"]) / self.n_classes
        }
        return metrics

def calculate_batch_metrics(y_true, y_pred, num_classes):
    """
    calculate metrics for one batch
    """
    y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    batch_tp = torch.zeros(num_classes, dtype=torch.float32)
    batch_fp = torch.zeros(num_classes, dtype=torch.float32)
    batch_fn = torch.zeros(num_classes, dtype=torch.float32)
    batch_tn = torch.zeros(num_classes, dtype=torch.float32)

    for cls in range(num_classes):
        batch_tp[cls] = torch.sum((y_pred == cls) & (y_true == cls)).item()
        batch_fp[cls] = torch.sum((y_pred == cls) & (y_true != cls)).item()
        batch_fn[cls] = torch.sum((y_pred != cls) & (y_true == cls)).item()
        batch_tn[cls] = torch.sum((y_pred != cls) & (y_true != cls)).item()

    return batch_tp, batch_fp, batch_fn, batch_tn


def calculate_epoch_metrics(tp, fp, fn, tn):
    """
    calculate metrics for one epoch
    """
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-7)

    # calculate metrics for each class and average them
    metrics = {
        "accuracy": np.mean(accuracy),
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "f1_score": np.mean(f1_score),
        "iou": np.mean(iou)
    }
    return metrics
