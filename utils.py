import torch
import torch.nn as nn
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
# Peak Signal-to-noise Ratio (PSNR) formula: 10 * log10(max_pixel^2 / MSE)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
# Unsigned int
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Ray helpers
def get_rays(H, W, K, c2w):
    """
    Ray Generator

    :param H: Height of the image
    :param W: Width of the image
    :param K: K =   [focal, 0,      0.5*W]
                    [0,     focal,  0.5*H]
                    [0,     0,      1]
    :param c2w: camera to world transform matrix

    :return:
        ray_origin: Origin of rays. Shape: [Height, Width, 3]
        ray_dir: Direction of rays. Shape: [Height, Width, 3]
    """
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H), indexing='xy')

    dirs = torch.stack([(i - K[0][2]) / K[0][0],
                        -(j - K[1][2]) / K[1][1],
                        -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = dirs @ (c2w[:3, :3].t())

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """
    Ray Generator

    :param H: Height of the image
    :param W: Width of the image
    :param K: K =   [focal, 0,      0.5*W]
                    [0,     focal,  0.5*H]
                    [0,     0,      1]
    :param c2w: camera to world transform matrix

    :return:
        ray_origin: Origin of rays. Shape: [Height, Width, 3]
        ray_dir: Direction of rays. Shape: [Height, Width, 3]
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = dirs @ (np.array(c2w[:3, :3]).transpose())

    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(z_vals_mid, weights, N_pts_fine, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans [N_rays, N_pts_coarse - 2]

    # Calculate Probability Density values (divide by sum to make sure that the sum is equal to 1)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # [N_rays, N_pts_coarse - 2]

    # Calculate Cumulative Distribution values
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (N_rays, N_pts_coarse - 1)

    # Take uniform samples
    if det:  # if perturb == 0. (no perturb)
        u = torch.linspace(0., 1., steps=N_pts_fine)
        u = u.expand(list(cdf.shape[:-1]) + [N_pts_fine])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_pts_fine])

    # Invert CDF
    u = u.contiguous()  # (N_rays, N_pts_fine)  refer to: https://stackoverflow.com/questions/48915810/

    # Move uniform or linspace points to closest cdf points (value -> index)
    # Hierarchical volume sampling but try to sample N_pts_fine points from N_pts_coarse-1 points, 
    # which will cause repeatingly sampling at same points.
    inds = torch.searchsorted(cdf, u, right=True)  # (N_rays, N_pts_fine)

    # Clip index to ensure 0 <= indx <= N_pts_coarse (below + 1 = above)
    below = torch.max(torch.zeros_like(inds), inds - 1)  # (N_rays, N_pts_fine)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)  # (N_rays, N_pts_fine)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_pts_fine, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # [N_rays, N_pts_fine, N_pts_coarse - 1]

    # Compute cumulated density values at points 'below' and 'above' (index -> value)
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)  # [N_rays, N_pts_fine, 2]

    # Compute middle sampling points of z_vals_mid at points 'below' and 'above' (index -> value)
    bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)  # [N_rays, N_pts_fine, 2]

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom  # [N_rays, N_pts_fine]
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # [N_rays, N_pts_fine]

    return samples
