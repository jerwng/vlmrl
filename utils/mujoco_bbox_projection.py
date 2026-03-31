#!/usr/bin/env python3
"""
MuJoCo 3D AABB computation and camera projection utilities for ground-truth
bounding box annotation.

No side effects: no file I/O, no printing except geom-type warnings.
Only imports: mujoco, numpy.
"""
import numpy as np
import mujoco

# MuJoCo geom type constants
GEOM_BOX      = mujoco.mjtGeom.mjGEOM_BOX
GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER

# Known object class prefixes and their display labels.
# Order matters for prefix-matching: longer prefixes first to avoid
# "cardboard_box" being matched by a hypothetical "cardboard" prefix.
OBJECT_PREFIXES = {
    "cardboard_box": "cardboard box",
    "trash_can":     "trash can",
    "bookshelf":     "bookshelf",
    "chair":         "chair",
    "table":         "table",
    "couch":         "couch",
}

# Fixed annotation color palette (RGB) per label, matching draw_multi_bbox.py.
LABEL_COLORS = {
    "chair":         (255, 0,   0),
    "table":         (0,   255, 0),
    "couch":         (0,   0,   255),
    "bookshelf":     (255, 255, 0),
    "cardboard box": (255, 0,   255),
    "trash can":     (0,   255, 255),
}


def get_camera_id(model: mujoco.MjModel, camera_name: str) -> int:
    """Return the integer camera index for camera_name. Raises ValueError if not found."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(
            f"Camera '{camera_name}' not found in model. "
            f"Available cameras: {[model.camera(i).name for i in range(model.ncam)]}"
        )
    return cam_id


def identify_object_bodies(model: mujoco.MjModel) -> list:
    """
    Scan all bodies in model and return those matching known room object naming conventions.

    Returns:
        List of (body_id, body_name, label) tuples.
        body_name matches '<prefix>_target' or '<prefix>_distractor<N>'.
        label is the human-readable class name (e.g. 'bookshelf', 'cardboard box').
    """
    results = []
    for body_id in range(model.nbody):
        body_name = model.body(body_id).name
        matched_label = None
        for prefix, label in OBJECT_PREFIXES.items():
            # Accept '<prefix>_target', '<prefix>_distractor<digits>', or '<prefix>_<digits>'
            if body_name == f"{prefix}_target":
                matched_label = label
                break
            if body_name.startswith(f"{prefix}_distractor"):
                suffix = body_name[len(f"{prefix}_distractor"):]
                if suffix.isdigit():
                    matched_label = label
                    break
            if body_name.startswith(f"{prefix}_"):
                suffix = body_name[len(f"{prefix}_"):]
                if suffix.isdigit():
                    matched_label = label
                    break
        if matched_label is not None:
            results.append((body_id, body_name, matched_label))
    return results


def _get_subtree_body_ids(model: mujoco.MjModel, root_id: int) -> list:
    """Return all body IDs in the subtree rooted at root_id (inclusive)."""
    result = [root_id]
    for i in range(model.nbody):
        if model.body_parentid[i] == root_id:
            result.extend(_get_subtree_body_ids(model, i))
    return result


def compute_body_aabb_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> np.ndarray:
    """
    Compute the 8 corners of the 3D axis-aligned bounding box (AABB) enclosing
    all geoms in the subtree rooted at body_id, in world frame coordinates.

    Object body XMLs (bookshelf.xml, chair.xml, etc.) define geoms on an unnamed
    child body rather than on the named root body. This function searches the entire
    subtree so that geoms on child bodies are captured correctly.

    Box and cylinder geoms are handled precisely; other types produce a warning
    and contribute a single point (their world position).
    Cylinders are bounded by a box of half-extents (radius, radius, half_length).

    Returns:
        corners: (8, 3) float64 array of AABB corner coordinates in world frame.
                 Returns a (1, 3) array with the body's world position if no geoms
                 are found in the subtree.
    """
    # Find all geom IDs in the subtree rooted at body_id
    subtree_ids = set(_get_subtree_body_ids(model, body_id))
    geom_ids = [
        gid for gid in range(model.ngeom)
        if model.geom_bodyid[gid] in subtree_ids
    ]

    if not geom_ids:
        # No geoms: return body xpos as a single point
        body_pos = data.xpos[body_id].copy()
        return body_pos.reshape(1, 3)

    all_points = []  # list of (N, 3) arrays; will be stacked

    for gid in geom_ids:
        geom_type = model.geom_type[gid]
        geom_xpos = data.geom_xpos[gid]      # world position (3,)
        geom_xmat = data.geom_xmat[gid].reshape(3, 3)  # world rotation matrix (3x3)

        if geom_type == GEOM_BOX:
            # model.geom_size[gid] holds [half_x, half_y, half_z] in geom local frame
            hx, hy, hz = model.geom_size[gid]
            local_corners = np.array([
                [ hx,  hy,  hz], [ hx,  hy, -hz],
                [ hx, -hy,  hz], [ hx, -hy, -hz],
                [-hx,  hy,  hz], [-hx,  hy, -hz],
                [-hx, -hy,  hz], [-hx, -hy, -hz],
            ], dtype=np.float64)  # (8, 3)
            world_corners = (geom_xmat @ local_corners.T).T + geom_xpos
            all_points.append(world_corners)
        elif geom_type == GEOM_CYLINDER:
            # model.geom_size[gid] = [radius, half_length, 0]
            # Cylinder axis is local Z; bound it with a box of half-extents (r, r, h).
            r, h = model.geom_size[gid][0], model.geom_size[gid][1]
            local_corners = np.array([
                [ r,  r,  h], [ r,  r, -h],
                [ r, -r,  h], [ r, -r, -h],
                [-r,  r,  h], [-r,  r, -h],
                [-r, -r,  h], [-r, -r, -h],
            ], dtype=np.float64)
            world_corners = (geom_xmat @ local_corners.T).T + geom_xpos
            all_points.append(world_corners)
        else:
            geom_type_name = mujoco.mjtGeom(geom_type).name
            print(
                f"[WARNING] Body {model.body(body_id).name}: geom {model.geom(gid).name} "
                f"has type '{geom_type_name}' (not box). Using world position as point estimate."
            )
            all_points.append(geom_xpos.reshape(1, 3))

    # Stack all points and compute AABB
    all_pts = np.vstack(all_points)   # (N, 3)
    mins = all_pts.min(axis=0)        # (3,)
    maxs = all_pts.max(axis=0)        # (3,)

    # Return 8 corners of the union AABB
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]],
        [maxs[0], maxs[1], maxs[2]],
    ], dtype=np.float64)
    return corners


def project_aabb_to_bbox2d(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_id: int,
    corners_world: np.ndarray,
    image_width: int = 336,
    image_height: int = 336,
) -> dict:
    """
    Project 3D AABB corners through the MuJoCo camera model to a 2D pixel bounding box.

    MuJoCo camera convention:
      - Camera looks along -Z in its local frame (cam_xmat columns are [right, up, -forward])
      - cam_xmat is stored row-major: cam_xmat.reshape(3,3) has rows = [right, up, -forward]
      - A point is in front of the camera iff P_cam[2] < 0

    Args:
        corners_world: (N, 3) float64 array of 3D corners in world frame.
        image_width, image_height: pixel dimensions of the rendered image.

    Returns:
        None if no corners project in front of the camera, or if the clamped
        bbox has zero width or zero height.
        Otherwise: {"bbox": [x, y, w, h], "truncated": bool}
          x, y: top-left pixel coordinate (integers)
          w, h: width and height in pixels (integers, >= 1)
          truncated: True if the pre-clamped bbox extended beyond image bounds.
    """
    # Retrieve camera pose from MjData (updated after mj_forward)
    t_cam = data.cam_xpos[cam_id]                   # (3,) world position
    R_cam = data.cam_xmat[cam_id].reshape(3, 3)     # (3,3) rotation matrix
    # MuJoCo cam_xmat convention: columns are the camera's local axes in world frame.
    #   col0 = right axis,  col1 = up axis,  col2 = -forward axis (camera looks along -Z local).
    # To transform world -> camera: P_cam = R_cam.T @ (P_world - t_cam)
    # A point is in front of the camera iff P_cam[2] < 0.

    fovy_deg = model.cam_fovy[cam_id]               # vertical FOV in degrees
    fovy_rad = np.deg2rad(fovy_deg)
    # Focal length in pixels (from vertical FOV)
    f = image_height / (2.0 * np.tan(fovy_rad / 2.0))

    # Transform all corners to camera frame
    # corners_world: (N, 3), t_cam: (3,)
    diff = corners_world - t_cam[np.newaxis, :]    # (N, 3)
    # P_cam = R_cam.T @ diff.T -> (3, N), then transpose -> (N, 3)
    P_cam = (R_cam.T @ diff.T).T                   # (N, 3)

    # Keep only points in front of camera (negative Z in camera frame)
    front_mask = P_cam[:, 2] < 0
    if not np.any(front_mask):
        return None

    P_front = P_cam[front_mask]                    # (M, 3), M >= 1

    # Perspective projection:
    # x_ndc = P_cam[0] / (-P_cam[2])
    # y_ndc = P_cam[1] / (-P_cam[2])
    # u = width/2  + f * x_ndc   (pixel x, left=0)
    # v = height/2 - f * y_ndc   (pixel y, top=0, Y-down)
    neg_z = -P_front[:, 2]                         # (M,), positive
    x_ndc = P_front[:, 0] / neg_z                  # (M,)
    y_ndc = P_front[:, 1] / neg_z                  # (M,)

    u = image_width  / 2.0 + f * x_ndc             # pixel x (M,)
    v = image_height / 2.0 - f * y_ndc             # pixel y (M,)

    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())

    # Detect truncation before clamping
    truncated = (
        u_min < 0 or u_max > image_width or
        v_min < 0 or v_max > image_height
    )

    # Clamp to image bounds
    u_min_c = max(0.0, u_min)
    u_max_c = min(float(image_width),  u_max)
    v_min_c = max(0.0, v_min)
    v_max_c = min(float(image_height), v_max)

    x = int(round(u_min_c))
    y = int(round(v_min_c))
    w = int(round(u_max_c)) - x
    h = int(round(v_max_c)) - y

    if w <= 0 or h <= 0:
        return None

    return {"bbox": [x, y, w, h], "truncated": truncated}
