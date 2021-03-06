import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, reduce=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.reduce = reduce

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def BCEWithLogitsLoss(self, logit, target):
        """This loss combines a Sigmoid layer and the BCELoss
        logit: a list of multiple input
        target: a list of gt
        """
        criterion = nn.BCEWithLogitsLoss(weight=self.weight, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()
        n = len(logit)
        loss = 0
        for i in range(n):
            loss += criterion(logit[i], target[i])
        return loss

    def CrossEntropyLoss(self, logit, target):
        # n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        # if self.batch_average:
        #    loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        # n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        # if self.batch_average:
        #    loss /= n

        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

    logit = torch.rand(1, 10, 10).cuda()
    logits = [logit, logit]
    target = torch.rand(1, 10, 10).cuda()
    targets = [target, target]
    print(loss.BCEWithLogitsLoss(logits, targets).item())