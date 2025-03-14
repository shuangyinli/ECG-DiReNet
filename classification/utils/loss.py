import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLossLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        """
        Initialize the BCEWithLogitsLoss with label smoothing.

        Args:
            smoothing (float): The label smoothing factor.
            pos_weight (torch.Tensor, optional): A weight of positive examples.
        """
        super(BCEWithLogitsLossLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        """
        Forward pass of the loss function.

        Args:
            logits (torch.Tensor): The raw, unnormalized predictions from the model.
            targets (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        targets_smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smoothed)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Initialize the Dice Loss.

        Args:
            smooth (float): A small value added to the denominator to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Forward pass of the Dice Loss.

        Args:
            inputs (torch.Tensor): The model predictions.
            targets (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed Dice Loss.
        """
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Initialize the Focal Loss.

        Args:
            alpha (float): The balancing factor.
            gamma (float): The focusing parameter.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.

        Args:
            inputs (torch.Tensor): The model predictions.
            targets (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed Focal Loss.
        """
        # Apply Sigmoid activation
        inputs = torch.sigmoid(inputs)
        
        # Calculate Binary Cross Entropy loss
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate Focal Loss
        p_t = torch.exp(-BCE_loss)  # p_t = exp(-BCE)
        loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, class_weights=None):
        """
        Cross Entropy Loss with label smoothing (for multi - class classification).

        Args:
            smoothing (float): The label smoothing coefficient (0~0.3).
            class_weights (torch.Tensor): The class weights tensor([w0, w1]).
        """
        super().__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights

    def forward(self, logits, targets):
        """
        Forward pass of the Cross Entropy Loss with label smoothing.

        Args:
            logits (torch.Tensor): The raw, unnormalized model output of shape [batch, 2].
            targets (torch.Tensor): The true labels, either of shape [batch] or [batch, 2] (one - hot).

        Returns:
            torch.Tensor: The computed loss.
        """
        num_classes = logits.size(-1)
        targets = targets.long() 
        # Convert targets to one - hot
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes).float()
        
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + self.smoothing / num_classes
        # Calculate cross entropy
        loss = -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1)

        # Apply class weights
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)
            weight_mask = torch.sum(targets * class_weights, dim=-1)
            loss = loss * weight_mask
        
        return loss.mean()

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean', eps=1e-8):
        """
        Initialize the Multi - Class Focal Loss.

        Args:
            alpha (torch.Tensor, optional): The class balancing factor.
            gamma (float): The focusing parameter.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            eps (float): A small value for numerical stability.
        """
        super().__init__()
        self.register_buffer('alpha', alpha)  # Optimization: register alpha as a buffer to ensure device synchronization
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps  # Protection for numerical stability

    def forward(self, logits, targets):
        """
        Forward pass of the Multi - Class Focal Loss.

        Args:
            logits (torch.Tensor): The raw, unnormalized model output.
            targets (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed Multi - Class Focal Loss.
        """
        device = logits.device
        targets = targets.long().to(device)

        log_p = F.log_softmax(logits, dim=-1)
        ce_loss = -log_p.gather(1, targets.view(-1, 1)).squeeze()  

        p_t = torch.exp(-ce_loss).clamp(min=self.eps, max=1-self.eps)  # Prevent probability from being 0 or 1

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
        """
        Apply the specified reduction to the loss.

        Args:
            loss (torch.Tensor): The computed loss before reduction.

        Returns:
            torch.Tensor: The reduced loss.
        """
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Multi - Class Dice Loss (for [batch, 2] output).

        Args:
            smooth (float): A small value added to the denominator to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Forward pass of the Multi - Class Dice Loss.

        Args:
            logits (torch.Tensor): The raw, unnormalized model output.
            targets (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed Multi - Class Dice Loss.
        """
        probs = F.softmax(logits, dim=-1)
        num_classes = probs.size(-1)
        
        # Convert targets to one - hot
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes).float()
        
        # Calculate Dice for each class
        intersection = torch.sum(probs * targets, dim=0)
        union = torch.sum(probs + targets, dim=0)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()
    

# Usage example --------------------------------------------------
if __name__ == "__main__":
    batch_size = 32
    num_classes = 2
    
    # Simulate data
    logits = torch.randn(batch_size, num_classes)  # Model output
    targets = torch.randint(0, num_classes, (batch_size,))  # True labels
    
    # Initialize loss functions
    ce_smooth_loss = CrossEntropyLabelSmoothing(smoothing=0.1)
    focal_loss = MultiClassFocalLoss(alpha=torch.tensor([1.0, 2.0]))  # Assume class 1 weight is 2.0
    dice_loss = MultiClassDiceLoss()
    
    # Calculate losses
    loss1 = ce_smooth_loss(logits, targets)
    loss2 = focal_loss(logits, targets)
    loss3 = dice_loss(logits, targets)
    
    # Example of combining losses
    total_loss = 0.5 * loss1 + 0.3 * loss2 + 0.2 * loss3
    
    print(f"CrossEntropy with Smoothing: {loss1.item():.4f}")
    print(f"Focal Loss: {loss2.item():.4f}")
    print(f"Dice Loss: {loss3.item():.4f}")
    print(f"Combined Loss: {total_loss.item():.4f}")