import torch
import torch.nn.functional as F
from torch.autograd import grad

# Photometric Loss: SSIM + L1
def photometric_loss(pred, target, alpha=0.85):
    # Structural Similarity (SSIM) Index
    def ssim(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim_map = SSIM_n / SSIM_d
        return torch.clamp((1 - ssim_map) / 2, 0, 1)

    ssim_loss = ssim(pred, target)
    l1_loss = torch.abs(pred - target)

    return alpha * ssim_loss.mean() + (1 - alpha) * l1_loss.mean()

# Regularization Loss for Depth and Flow
def regularization_loss(depth, image):
    # Compute image gradients (assuming grayscale)
    def gradient_x(img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    # Gradients for predicted depth and image
    d_x = gradient_x(depth)
    d_y = gradient_y(depth)
    
    img_x = gradient_x(image)
    img_y = gradient_y(image)

    # Regularization loss (smoothness) weighted by the image gradients
    reg_loss_x = torch.abs(d_x) * torch.exp(-torch.abs(img_x))
    reg_loss_y = torch.abs(d_y) * torch.exp(-torch.abs(img_y))

    return reg_loss_x.mean() + reg_loss_y.mean()

# Total Loss for Depth and Flow
def compute_total_loss(pred_depth, pred_flow, images, target_depths, alpha=0.85):
    # Photometric loss for depth (using ego-motion warped images)
    loss_photo_depth = photometric_loss(pred_depth, images, alpha)

    # Regularization loss for depth
    loss_reg_depth = regularization_loss(pred_depth, images)

    # Photometric loss for optical flow
    loss_photo_flow = photometric_loss(pred_flow, images, alpha)

    # Regularization loss for optical flow
    loss_reg_flow = regularization_loss(pred_flow, images)

    # Total loss (weighted sum of all losses)
    total_loss = (
        loss_photo_depth + loss_reg_depth + loss_photo_flow + loss_reg_flow
    )
    return total_loss
