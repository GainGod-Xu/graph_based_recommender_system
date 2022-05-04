import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.alpha = alpha
        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        # ce = -(target * torch.log(nn.functional.softmax(input)))
        # pt = torch.exp(-ce)
        # loss = (1 - pt) ** self.gamma * ce

        # if self.alpha >= 0:
        #     alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        #     loss = alpha_t * loss
        bce_l = self.bceloss(input, target)
        # print(bce_l.min(), bce_l.max())

        pt = torch.clamp(torch.exp(-bce_l), min=1e-6, max=1-1e-6)
        loss = (torch.pow(1-pt, self.gamma) * bce_l)
        # print(loss.min(), loss.max())
        return loss.sum()



# import torch.nn as nn
# import torch
# import numpy as np
# import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=-1, gamma=0):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.alpha = alpha

#     def forward(self, inputs, targets):
#         """
#         Implement forward of focal loss
#         :param input: input predictions
#         :param target: labels
#         :return: tensor of focal loss in scalar
#         """
#         p = torch.sigmoid(inputs)
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#         p_t = p * targets + (1 - p) * (1 - targets)
#         # print(p_t)
#         # p_t = torch.clamp(p_t, min=1e-6, max=1-1e-6)
#         loss = ce_loss * ((1 - p_t) ** self.gamma)
#         # loss = torch.clamp(loss, min=1e-6, max=1-1e-6)

#         if self.alpha >= 0:
#             alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
#             loss = alpha_t * loss

#         return loss.mean()
