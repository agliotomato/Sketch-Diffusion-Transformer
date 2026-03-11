import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import lpips
from torchvision.transforms.functional import gaussian_blur


class HybridLoss(nn.Module):
    """
    Multi-scale Shape + Gradient + LPIPS loss for hair generation.
    Returns three separate loss components; weighting is handled by the caller.
    """

    SHAPE_KERNELS  = [11, 21, 31]
    SOBEL_SCALES   = [3, 5, 7]
    MASK_BLUR_K    = 61
    MASK_BLUR_SIGMA = 10.0

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

        print("Loading LPIPS VGG model...")
        self.lpips_net = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips_net.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def soft_mask(self, mask):
        """Blur a latent-space mask for soft blending. mask: (B,1,H,W)"""
        return gaussian_blur(mask, [self.MASK_BLUR_K, self.MASK_BLUR_K], self.MASK_BLUR_SIGMA)

    # ------------------------------------------------------------------
    # Loss components (pixel space)
    # ------------------------------------------------------------------

    def _shape_loss(self, pred, gt, mask):
        """Multi-scale Gaussian L1 (shape / silhouette)."""
        total = 0.0
        for k in self.SHAPE_KERNELS:
            sigma = k / 4.0
            p_blur = gaussian_blur(pred, [k, k], sigma)
            g_blur = gaussian_blur(gt,   [k, k], sigma)
            diff = torch.abs(p_blur - g_blur)
            total += _masked_mean(diff, mask)
        return total / len(self.SHAPE_KERNELS)

    def _gradient_loss(self, pred, gt, mask):
        """Multi-scale Sobel L1 (thin edges / braid lines)."""
        total = 0.0
        for k in self.SOBEL_SCALES:
            if k > 3:
                sigma = (k - 1) / 4.0
                p = gaussian_blur(pred, [k, k], sigma)
                g = gaussian_blur(gt,   [k, k], sigma)
            else:
                p, g = pred, gt
            diff = torch.abs(kornia.filters.sobel(p) - kornia.filters.sobel(g))
            total += _masked_mean(diff, mask)
        return total / len(self.SOBEL_SCALES)

    def _lpips_loss(self, pred, gt):
        return self.lpips_net(pred, gt).mean()

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, v_pred, z0, z_t, sigmas, mask):
        """
        Args:
            v_pred : (B, 16, H, W) model velocity prediction
            z0     : (B, 16, H, W) clean latents
            z_t    : (B, 16, H, W) noisy latents
            sigmas : (B, 1, 1, 1)  noise level
            mask   : (B, 1, 1024, 1024) original-resolution matte [0,1]
        Returns:
            loss_shape, loss_gradient, loss_lpips   (all scalar tensors)
        """
        # Reconstruct predicted clean latent
        z0_pred = z_t - sigmas * v_pred

        with torch.cuda.amp.autocast(enabled=False):
            pixel_pred = self.vae.decode(z0_pred.float() / self.vae.config.scaling_factor, return_dict=False)[0]
            pixel_gt   = self.vae.decode(z0.float()      / self.vae.config.scaling_factor, return_dict=False)[0]

        pixel_mask = F.interpolate(mask, size=pixel_pred.shape[-2:], mode="nearest")

        loss_shape    = self._shape_loss(pixel_pred.float(), pixel_gt.float(), pixel_mask.float())
        loss_gradient = self._gradient_loss(pixel_pred.float(), pixel_gt.float(), pixel_mask.float())
        loss_lpips    = self._lpips_loss(pixel_pred.float(), pixel_gt.float())

        return loss_shape, loss_gradient, loss_lpips


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _masked_mean(diff, mask):
    """Weighted mean over masked region. diff and mask same spatial dims."""
    return (diff * mask).sum() / (mask.sum() * diff.shape[1] + 1e-6)


def time_weights(timesteps, lambda_shape, lambda_gradient, lambda_lpips, sigmas):
    """
    Returns per-batch scalar weights for the three loss terms.
    Curriculum: layout(t>700) → structure(300<t<700) → texture(t<300)
    """
    device = timesteps.device
    w_shape    = torch.full((len(timesteps),), lambda_shape,    device=device)
    w_gradient = torch.full((len(timesteps),), lambda_gradient, device=device)

    # Layout phase: boost shape
    mask_layout = timesteps >= 700
    w_shape = torch.where(mask_layout, w_shape * 1.5, w_shape)

    # Structure phase: boost gradient
    mask_struct = (timesteps >= 300) & (timesteps < 700)
    w_gradient = torch.where(mask_struct, w_gradient * 3.0, w_gradient)

    # Texture phase: exponential LPIPS boost at low noise
    s = sigmas.flatten().float()  # (B,)
    w_lpips = lambda_lpips * 5.0 * torch.exp(-4.0 * s)

    return w_shape.mean(), w_gradient.mean(), w_lpips.mean()
