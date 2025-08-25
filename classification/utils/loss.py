import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLossLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super(BCEWithLogitsLossLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        targets_smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smoothed)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        targets = targets.long()
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes).float()
        targets = targets * (1 - self.smoothing) + self.smoothing / num_classes
        loss = -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1)
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            weight_mask = torch.sum(targets * class_weights, dim=-1)
            loss = loss * weight_mask
        return loss.mean()


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean', eps=1e-8):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        device = logits.device
        targets = targets.long().to(device)

        log_p = F.log_softmax(logits, dim=-1)
        ce_loss = -log_p.gather(1, targets.view(-1, 1)).squeeze()  
        p_t = torch.exp(-ce_loss).clamp(min=self.eps, max=1-self.eps)

        if self.alpha is not None:
            self.alpha = self.alpha.to(device)
            alpha = self.alpha.gather(0, targets.view(-1)) 
            focal_loss = alpha * (1 - p_t)**self.gamma * ce_loss
        else:
            focal_loss = (1 - p_t)**self.gamma * ce_loss

        if torch.isnan(focal_loss).any():
            raise RuntimeError("NaN detected in focal loss")
        return self._reduce_loss(focal_loss)

    def _reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        num_classes = probs.size(-1)
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes).float()
        intersection = torch.sum(probs * targets, dim=0)
        union = torch.sum(probs + targets, dim=0)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    

if __name__ == "__main__":
    batch_size = 32
    num_classes = 2
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    ce_smooth_loss = CrossEntropyLabelSmoothing(smoothing=0.1)
    focal_loss = MultiClassFocalLoss(alpha=torch.tensor([1.0, 2.0]))
    dice_loss = MultiClassDiceLoss()
    
    loss1 = ce_smooth_loss(logits, targets)
    loss2 = focal_loss(logits, targets)
    loss3 = dice_loss(logits, targets)
    
    total_loss = 0.5 * loss1 + 0.3 * loss2 + 0.2 * loss3
    
    print(f"CrossEntropy with Smoothing: {loss1.item():.4f}")
    print(f"Focal Loss: {loss2.item():.4f}")
    print(f"Dice Loss: {loss3.item():.4f}")
    print(f"Combined Loss: {total_loss.item():.4f}")
