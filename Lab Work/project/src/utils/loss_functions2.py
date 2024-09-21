import torch
import torch.nn.functional as F

# Photometric Loss: SSIM + L1
def photometric_loss(pred, target, alpha=0.85):
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
def regularization_loss(pred, image):
    def gradient_x(img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    pred_x = gradient_x(pred)
    pred_y = gradient_y(pred)
    
    img_x = gradient_x(image)
    img_y = gradient_y(image)

    reg_loss_x = torch.abs(pred_x) * torch.exp(-torch.abs(img_x))
    reg_loss_y = torch.abs(pred_y) * torch.exp(-torch.abs(img_y))

    return reg_loss_x.mean() + reg_loss_y.mean()


def warp_image_with_depth(img, depth, ego_motion, intrinsic_matrix):
    """
    Warps the source image using the predicted depth and ego-motion.
    """
    # Apply depth and ego-motion warping here (not fully implemented for brevity)
    # You would typically apply a 3D transformation based on depth and motion
    return img  # This would return the warped image based on the projection

def flow_warp(img, flow):
    """
    Warps the source image using the predicted optical flow.
    """
    B, C, H, W = img.shape
    i_range = torch.arange(0, W).view(1, 1, -1).expand(B, H, W).to(img.device)
    j_range = torch.arange(0, H).view(1, -1, 1).expand(B, H, W).to(img.device)

    flow_u = flow[:, 0, :, :]
    flow_v = flow[:, 1, :, :]

    i_range = i_range.float() + flow_u
    j_range = j_range.float() + flow_v

    i_range = 2.0 * i_range / (W - 1) - 1.0
    j_range = 2.0 * j_range / (H - 1) - 1.0

    grid = torch.stack([i_range, j_range], dim=-1).permute(0, 2, 3, 1)
    warped_img = F.grid_sample(img, grid, align_corners=True)
    return warped_img


def compute_total_loss(pred_depth, pred_flow, images, target_images, ego_motion, intrinsic_matrix, alpha=0.85):
    # 1. Photometric loss for depth (use ego-motion and depth to warp image)
    warped_img_depth = warp_image_with_depth(images, pred_depth, ego_motion, intrinsic_matrix)
    loss_photo_depth = photometric_loss(warped_img_depth, target_images, alpha)

    # 2. Regularization loss for depth (smoothness)
    loss_reg_depth = regularization_loss(pred_depth, images)

    # 3. Photometric loss for optical flow (warp image using flow)
    warped_img_flow = flow_warp(images, pred_flow)
    loss_photo_flow = photometric_loss(warped_img_flow, target_images, alpha)

    # 4. Regularization loss for flow (smoothness)
    loss_reg_flow = regularization_loss(pred_flow, images)

    # Total loss (sum of all components)
    total_loss = loss_photo_depth + loss_reg_depth + loss_photo_flow + loss_reg_flow
    return total_loss
