import torch


class NucleusLoss(torch.nn.Module):

    def __init__(self, alpha=2, dice_beta=1):
        super(NucleusLoss, self).__init__()
        self.alpha = alpha
        self.dice_beta_square = dice_beta * dice_beta

    def forward(self, logits, labels, epsilon=1e-7):
        masks = labels.float()
        batch_size = logits.size(0)
        weights = 1 + self.alpha * torch.abs(torch.nn.functional.avg_pool2d(
            masks, kernel_size=31, stride=1, padding=15) - masks)
        weights = weights.view(batch_size, -1)
        logits = logits.view(batch_size, 2, -1)
        labels = labels.view(batch_size, -1)
        masks = masks.view(batch_size, -1)

        CE_loss = torch.nn.functional.cross_entropy(
            logits, labels, reduction='none')
        CE_loss = (weights * CE_loss).sum(-1) / weights.sum(-1)

        pred_masks = torch.softmax(logits, 1).select(1, 1).squeeze(1)
        intersection = (pred_masks * masks * weights).sum(-1)
        union = (
            (pred_masks + self.dice_beta_square * masks) * weights).sum(-1)
        Dice = (intersection + epsilon) / (union + epsilon)
        Dice = 0.5 * (1 + self.dice_beta_square) * Dice
        return (CE_loss - Dice).mean()


class TissueLoss(torch.nn.Module):

    def __init__(self, num_classes, label_weights, dice_beta=2):
        super(TissueLoss, self).__init__()
        label_weights = torch.Tensor(label_weights)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_weights)
        self.dice_loss = DiceLoss(num_classes, beta=dice_beta)

    def forward(self, input, target):
        loss = self.cross_entropy_loss(input, target)
        loss += self.dice_loss(input, target)
        return loss

    def cuda(self, device=None):
        self = super().cuda(device)
        self.cross_entropy_loss = self.cross_entropy_loss.cuda(device)
        self.dice_loss = self.dice_loss.cuda(device)
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.cross_entropy_loss = self.cross_entropy_loss.to(*args, **kwargs)
        self.dice_loss = self.dice_loss.to(*args, **kwargs)
        return self


class DiceLoss(torch.nn.Module):

    def __init__(self, num_classes, beta=1, epsilon=1e-7, device=None):
        super(DiceLoss, self).__init__()
        self.beta_square = beta * beta
        self.num_classes = num_classes
        weights = [0. if i == 0 else 1. for i in range(num_classes)]
        self.weights = torch.Tensor(weights)
        if device is not None:
            self.weights =  self.weights.to(device)
        self.epsilon = epsilon

    def forward(self, input, target):
        input = input.view(input.size(0), input.size(1), -1)
        input = input.transpose(1, 2)
        input = input.contiguous().view(-1, input.size(2))
        input = torch.softmax(input, dim=1)
        target = torch.nn.functional.one_hot(
            target.view(-1), num_classes=self.num_classes)
        intersection = (
            (1 + self.beta_square) * torch.sum(input * target, 0)
            + self.epsilon)
        cardinality = (
            torch.sum(input, 0) + self.beta_square * torch.sum(target, 0)
            + self.epsilon)
        dice_scores = self.weights * intersection / cardinality
        return - torch.mean(dice_scores)

    def cuda(self, device=None):
        self = super().cuda(device)
        self.weights = self.weights.cuda(device)
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.weights = self.weights.to(*args, **kwargs)
        return self
