from utils import *


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=nn.functional.relu: 1. - torch.exp(-act_fn(raw) * dists)

    # get intervals between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_pts_x]

    # Multiply each distance by the norm of its corresponding direction ray 
    # to convert to real world distance (accounts for non-unit directions).
    # refer to: https://github.com/bmild/nerf/issues/113
    dists = dists * torch.linalg.norm(rays_d[..., None, :], dim=-1)  # [N_rays, N_pts_x]

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_pts_x, 3]

    # Add noise to model's predictions for density. Can be used to 
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_pts_x]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]  # [N_rays, N_pts_x]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, network_query_fn, 
                network_fn, N_pts_coarse,
                network_fine=None, N_pts_fine=0,
                retraw=False, lindisp=False, perturb=0.,
                white_bkgd=False, raw_noise_std=0.,
                verbose=False):
    """Volumetric rendering.
    Args:
    render_kwargs_xx: go to nerf.py --> create_nerf() --> render_kwargs_train for reference
        1. network_query_fn: function used for passing queries to network_fn.
        2. network_fn: function. Model for predicting RGB and density at each point in space.
        3. N_pts_coarse: int. Number of different times to sample along each ray.
        4. network_fine: "fine" network with same spec as network_fn.
        5. N_pts_fine: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
        6. perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
        7. white_bkgd: bool. If True, assume a white background.
        8. raw_noise_std: ...
    ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, near, far, view direction.
    retraw: bool. If True, include model's raw, unprocessed predictions.
    lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
    verbose: bool. If True, print more debugging info.

    Returns:
    rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
    disp_map: [num_rays]. Disparity map. 1 / depth.
    acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
    raw: [num_rays, num_samples, 4]. Raw predictions from model.
    rgb_coarse: See rgb_map. Output for coarse model.
    disp_coarse: See disp_map. Output for coarse model.
    acc_coarse: See acc_map. Output for coarse model.
    z_std: [num_rays]. Standard deviation of distances along ray for each sample.
    """

    # Unpack rays_o(3), rays_d(3), near(1), far(1), viewdirs(3) in ray_batch
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [B, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [B, 1]

    # Create sampling distance over rays
    t_vals = torch.linspace(0., 1., steps=N_pts_coarse)  # [N_pts_coarse]
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)  # [B, N_pts_coarse]
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        z_vals = lower + (upper - lower) * t_rand

    # Create sampling points: [B, 1, 3] + [B, 1, 3] * [B, N_pts_coarse, 1]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [B, N_pts_coarse, 3]

    # Estimate color and density at each point using NeRF
    raw = network_query_fn(pts, viewdirs, network_fn)  # [B, N_pts_coarse, 4]

    # Volume rendering
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    # Coarse to fine strategy
    if N_pts_fine > 0:
        rgb_map_coarse, disp_map_coarse, acc_map_coarse = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_pts_fine, det=(perturb == 0.))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # [B, N_pts_coarse + N_pts_fine, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [B, N_pts_coarse + N_pts_fine, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)  # [B, N_pts_coarse + N_pts_fine, 4]

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    # Pack output
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_pts_fine > 0:
        ret['rgb_coarse'] = rgb_map_coarse
        ret['disp_coarse'] = disp_map_coarse
        ret['acc_coarse'] = acc_map_coarse
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [B]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and verbose:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    Note: chunk is larger than N_rand(number of ray batches), so it has no effect
    at training step. However, it helps when comes to render a full image(512x512)
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: array. Focal length of pinhole camera, H//2 and W//2.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [ro+rd(2), batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)  # [H, W, 3], [H, W, 3]
    else:
        # use provided ray batch
        rays_o, rays_d = rays  # [B, 3], [B, 3]

    # View dependency
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [B, 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])  # [B, 1], [B, 1]
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # [B, D_rays_o + D_rays_d + D_near + D_far(3+3+1+1=8)]
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)  # [B, D_rays_o + D_rays_d + D_near + D_far + D_viewdirs(3+3+1+1+3=11)]

    # Render and reshape
    all_ret = batchify(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]