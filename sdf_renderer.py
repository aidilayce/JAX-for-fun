# needed an sdf normal  & depth renderer, so here it is
# BEWARE: assumes inconventional sdf as -1 outside 1 inside

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
import math 

def render_normals(sdf,
                    cam_rot,
                    camera_origin=np.array([0., 0., 2.5]), 
                    bbox_min=np.array([-1.10, -1.10, -1.10]), 
                    bbox_max=np.array([1.10, 1.10, 1.10]),
                    img_size=(512,512), 
                    num_steps=100, 
                    thresh=1.0, 
                    step_scale=0.1):
    """
    Render differentiable normal map from SDF/TSDF.

    Parameters:
        sdf (torch.Tensor): SDF grid (N,N,N).
        cam_rot (np.ndarray): Camera rotation matrix (xyz).
        focal_length (float): Camera focal length.
        camera_origin (np.ndarray): Camera origin.
        bbox_min (np.ndarray): Minimum coordinates of the bounding box.
        bbox_max (np.ndarray): Maximum coordinates of the bounding box.
        img_size (tuple): Rendered image size (H,W).
        num_steps (int): Ray marching steps.
        thresh (float): Intersection threshold.
        step_scale (float): Step scaling factor.

    Returns:
        np.ndarray: Rendered normal map (H,W,3).
    """
    
    tsdf_grid = jnp.array(sdf.detach().cpu().numpy()) 

    grid_size = tsdf_grid.shape[0]

    # world coordinates to voxel indices
    def world_to_voxel(p):
        scale = (grid_size - 1) / (bbox_max - bbox_min)
        return (p - bbox_min) * scale

    # interpolation of TSDF values w/ jax map_coordinates. this is why this rendering is so fast! thnx jax
    def tsdf_interpolate(tsdf, points):
        coords = world_to_voxel(points).T
        interp = map_coordinates(tsdf, coords, order=1, mode='constant', cval=-1.0)
        # note: prev, this was nearest sampling as "map_coordinates(tsdf, coords, order=1, mode='nearest')"
        # but this created issues for the border sdf values, so changed setting to constant -1 (outside) value to ensure ways exit the volume
        return interp

    # get the gradient function to compute normals
    tsdf_grad_fn = jax.grad(lambda p: tsdf_interpolate(tsdf_grid, p).sum())

    # again, thnx jat so that we can do vectorized ray marching here
    @jax.jit
    def ray_march_single(ray_origin, ray_dir, tsdf):
        def cond_fn(state):
            pos, prev_sdf, sdf, step, hit = state
            crossed_surface = (prev_sdf < 0.0) & (sdf >= 0.0) & (jnp.abs(sdf) < thresh)
            return (~crossed_surface) & (step < num_steps)

        def body_fn(state):
            pos, prev_sdf, sdf, step, hit = state
            prev_sdf = sdf
            step_size = jnp.abs(sdf) * step_scale # jnp.clip(jnp.abs(sdf) * step_scale, 0.0, 0.02)
            pos = pos + ray_dir * step_size
            sdf = tsdf_interpolate(tsdf, pos[None,:])[0]
            hit = (prev_sdf < 0.0) & (sdf >= 0.0) & (jnp.abs(sdf) < thresh)
            return pos, prev_sdf, sdf, step+1, hit

        init_sdf = tsdf_interpolate(tsdf, ray_origin[None,:])[0]
        init_state = (ray_origin, init_sdf, init_sdf, 0, False)

        final_pos, _, _, _, hit = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return final_pos, hit

    # finally, vectorized render of the normal map
    @jax.jit
    def render(camera_origin, rays, tsdf):
        origins = jnp.tile(camera_origin, (rays.shape[0], 1))
        positions, hits = jax.vmap(ray_march_single, (0,0,None))(origins, rays, tsdf)
        normals = -jax.vmap(tsdf_grad_fn)(positions) # negative gradient is the normal
        normals = normals / (jnp.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
        return normals

    # compute rays
    H, W = img_size
    i, j = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    cx = (W - 1) / 2.
    cy = (H - 1) / 2.
    fov_deg = 49.1
    focal_length = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    x = (j - cx) / focal_length
    y = -(i - cy) / focal_length
    z = -jnp.ones_like(x)

    rays = jnp.stack([x, y, z], axis=-1)
    rays /= jnp.linalg.norm(rays, axis=-1, keepdims=True)
    rays_flat = rays.reshape(-1, 3)

    # rotate the rays
    rays_flat = jnp.dot(rays_flat, cam_rot.T)
    camera_origin = jnp.dot(np.array(camera_origin), cam_rot.T)
    # camera_origin = jnp.array(camera_origin)

    normal_map = render(camera_origin, rays_flat, tsdf_grid).reshape(H, W, 3)

    # normalize the normal map
    normalized_normal_map = (normal_map - normal_map.min()) / (normal_map.max() - normal_map.min())

    # make the background black
    normalized_normal_map = normalized_normal_map.at[normalized_normal_map==normalized_normal_map[0,0,0]].set(0.0)

    return np.array(normalized_normal_map)


import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
import math 

def render_depth(sdf,
                    cam_rot,
                    camera_origin=np.array([0., 0., 2.5]), 
                    bbox_min=np.array([-1.10, -1.10, -1.10]), 
                    bbox_max=np.array([1.10, 1.10, 1.10]),
                    fov_deg=49.1,
                    img_size=(512,512), 
                    num_steps=300, 
                    thresh=1.0, 
                    step_scale=0.1,
                    max_depth=1.0):
    """
    Render differentiable depth map from SDF/TSDF.
    BEWARE: This function assumes that inside of the SDF is 1 and outside is 0-1 (following Hunyuan3D-2).

    Parameters:
        sdf (torch.Tensor): SDF grid (N,N,N).
        cam_rot (np.ndarray): 3x3 rotation matrix for the camera (world ← camera, xyz).
        camera_origin (np.ndarray): Camera origin.
        bbox_min (np.ndarray): Minimum coordinates of the bounding box.
        bbox_max (np.ndarray): Maximum coordinates of the bounding box.
        fov_deg (float): Camera field of view in degrees.
        img_size (tuple): Rendered image size (H,W).
        num_steps (int): Ray marching steps.
        thresh (float): Intersection threshold.
        step_scale (float): Step scaling factor.
        max_depth (float): Maximum depth value for the depth map if a ray does not hit the surface.

    Returns:
        np.ndarray: Rendered depth map (H,W,1).
    """
    
    tsdf_grid = jnp.array(sdf.detach().cpu().numpy()) 

    grid_size = tsdf_grid.shape[0]

    # world coordinates to voxel indices
    def world_to_voxel(p):
        scale = (grid_size - 1) / (bbox_max - bbox_min)
        return (p - bbox_min) * scale

    # interpolation of TSDF values w/ jax map_coordinates. this is why this rendering is so fast! thnx jax
    def tsdf_interpolate(tsdf, points):
        coords = world_to_voxel(points).T
        interp = map_coordinates(tsdf, coords, order=1, mode='constant', cval=-1.0)
        # note: prev, this was nearest sampling as "map_coordinates(tsdf, coords, order=1, mode='nearest')"
        # but this created issues for the border sdf values, so changed setting to constant -1 (outside) value to ensure ways exit the volume
        return interp

    # get the gradient function to compute normals
    tsdf_grad_fn = jax.grad(lambda p: tsdf_interpolate(tsdf_grid, p).sum())

    # again, thnx jat so that we can do vectorized ray marching here
    @jax.jit
    def ray_march_single(ray_origin, ray_dir, tsdf):
        # outside to inside is smaller than 0 to bigger than 0 because
        # Hunyuan sdf is positive inside and negative outside
        def cond_fn(state):
            pos, prev_sdf, sdf, step, hit = state
            crossed_surface = (prev_sdf < 0.0) & (sdf >= 0.0) & (jnp.abs(sdf) < thresh)
            return (~crossed_surface) & (step < num_steps)

        def body_fn(state):
            pos, prev_sdf, sdf, step, hit = state
            prev_sdf = sdf
            step_size = jnp.abs(sdf) * step_scale # jnp.clip(jnp.abs(sdf) * step_scale, 0.0, 0.02)
            pos = pos + ray_dir * step_size
            sdf = tsdf_interpolate(tsdf, pos[None,:])[0]
            hit = (prev_sdf < 0.0) & (sdf >= 0.0) & (jnp.abs(sdf) < thresh)
            return pos, prev_sdf, sdf, step+1, hit

        init_sdf = tsdf_interpolate(tsdf, ray_origin[None,:])[0]
        init_state = (ray_origin, init_sdf, init_sdf, 0, False)

        final_pos, _, _, _, hit = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return final_pos, hit

    # finally, vectorized render of the normal map
    @jax.jit
    def render(camera_origin, rays, tsdf):
        origins = jnp.tile(camera_origin, (rays.shape[0], 1))
        positions, hits = jax.vmap(ray_march_single, (0,0,None))(origins, rays, tsdf)
        depths = jnp.linalg.norm(positions - camera_origin, axis=-1) # depth is the distance from the camera
        depths = jnp.where(hits, depths, max_depth)
        return depths

    # compute rays
    H, W = img_size
    i, j = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    cx = (W - 1) / 2.
    cy = (H - 1) / 2.
    focal_length = (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    x = (j - cx) / focal_length
    y = -(i - cy) / focal_length
    z = -jnp.ones_like(x)

    rays = jnp.stack([x, y, z], axis=-1)
    rays /= jnp.linalg.norm(rays, axis=-1, keepdims=True)
    rays_flat = rays.reshape(-1, 3)

    # rotate the rays
    rays_flat = jnp.dot(rays_flat, cam_rot.T)
    camera_origin = jnp.dot(np.array(camera_origin), cam_rot.T)
    
    # render the normal map
    depth_map = render(camera_origin, rays_flat, tsdf_grid).reshape(H, W, 1)

    # normalize the normal map
    normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # make the background black bcuz with the normalization it's gray
    depth_map = normalized_depth_map.at[normalized_depth_map==normalized_depth_map[0,0,0]].set(0.0)

    return np.array(depth_map)
                      
