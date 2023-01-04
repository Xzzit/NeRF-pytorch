import torch
import torch.nn as nn
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
# Peak Signal-to-noise Ratio (PSNR) formula: 10 * log10(max_pixel^2 / MSE)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
# Unsigned int
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    """
    Positional Encoding

    :param input_dims: the dimension of input data. Default: 3 (x, y, z).
    :param include_input: whether include input data into output. Default: True.
    :param max_freq_log2:  the maximum frequency.
    :param num_freqs: number of sampling frequency point.
    :param log_sampling: sampling through [0, max_freq], then 2^[sampling point]. Default: True.
    :param periodic_fns: periodic functions used for positional encoding.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns  # [sin(2^0)(), ..., cos(2^max_freq)()], shape: [B*H*W, 6*(max_freq-1)+3]
        self.out_dim = out_dim  # 6*(max_freq-1)+3

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    '''

    :param multires: number of frequency
    :param i: -1 for not positional embedding

    :return:
        embed:
        embedder_obj.out_dim: channel after positional embedding
    '''
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


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
def sample_pdf(bins, weights, N_pts_coarse, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_pts_coarse)
        u = u.expand(list(cdf.shape[:-1]) + [N_pts_coarse])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_pts_coarse])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_pts_coarse]
        if det:
            u = np.linspace(0., 1., N_pts_coarse)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_pts_coarse, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
