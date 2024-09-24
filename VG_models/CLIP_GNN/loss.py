import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p = F.softmax(inputs, dim=1)
        
        # Get the probabilities corresponding to the target labels
        p_t = p * targets_one_hot
        p_t = p_t.sum(dim=1)

        focal_factor = torch.pow((1 - p_t), self.gamma)

        loss = - focal_factor * torch.log(p_t)

        return loss.mean()


class MultiLabelCELoss(nn.Module):
    """
    BCEWithLogitsLoss with input logits and target as a list of labels
    """
    def __init__(self):
        super(MultiLabelCELoss, self).__init__()

    def convert_to_one_hot(self, logits, databatch):
        batch_size = len(databatch)
        n_objects_in_batch = databatch.x.shape[0]
        y_one_hot = torch.zeros((batch_size, n_objects_in_batch))
        shift = 0
        for i in range(batch_size):
            single_graph_data = databatch[i]
            n_objects_in_graph = single_graph_data.x.shape[0]
            y = single_graph_data.y
            y_one_hot[i, y + shift] = 1
            shift += n_objects_in_graph

        return y_one_hot
    
    def forward(self, logits, data):
        y_one_hot = self.convert_to_one_hot(logits, data)
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(logits, y_one_hot)