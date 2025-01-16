import torch
import torch.nn as nn

class MulticlassDiceLoss(nn.Module):
    """Reference: 1. https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
                  2.https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """
    def __init__(self, num_classes, softmax_dim=None):
        super().__init__()
        self.num_classes = num_classes - 1 # remove the final class
        self.softmax_dim = softmax_dim

    def forward(self, logits, targets, reduction='mean', smooth=1e-6):
        """This method computes the dice loss for all classes except the last one."""
        probabilities = logits
        if self.softmax_dim is not None:
            probabilities = nn.Softmax(dim=self.softmax_dim)(logits)
        
        # One-hot encode the target labels
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes + 1)
        # Convert from NHWC to NCHW
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        probabilities = probabilities[:, :self.num_classes, :, :]
        targets_one_hot = targets_one_hot[:, :self.num_classes, :, :]

        # Calculate intersection and union
        intersection = (targets_one_hot * probabilities).sum(dim=(0, 2, 3))
        mod_a = probabilities.sum(dim=(0, 2, 3))
        mod_b = targets_one_hot.sum(dim=(0, 2, 3))

        # Calculate Dice coefficient
        dice_coefficient = 2. * intersection / (mod_a + mod_b + smooth)
        dice_loss = -dice_coefficient.mean()
        
        return dice_loss
# end class MulticlassDiceLoss

if __name__ == '__main__':

    criterion = MulticlassDiceLoss(num_classes=3, softmax_dim=1)
    y = torch.randn(10, 3, 4, 4)
    ground_truth = torch.randint(0, 3, (10, 4, 4))
    print(y.shape)
    print(criterion(y, ground_truth))
