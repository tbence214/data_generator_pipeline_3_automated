import os
import queue
import random
from collections import Counter

import carla
import cv2
import numpy as np

from carla_config import SimulationConfig
from carla_utils import build_projection_matrix, ensure_directories, resolve_seeds

# ==========================================
# CARLA 0.9.15 SEMANTIC TAG CONSTANTS
# Verify by printing np.unique(semantic_map) for a frame if detections fail.
# ==========================================
TAG_VEHICLES    = (14,15,16,18,19)  # CityScapes label for vehicles
TAG_PEDESTRIANS = (12,)    # CityScapes label for walkers/pedestrians

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================

def get_coco_class(blueprint_id):
    if blueprint_id.startswith("walker."):
        return 0
    if blueprint_id.startswith("vehicle."):
        if any(x in blueprint_id for x in ["bicycle", "crossbike", "century", "omafiets"]):
            return 1
        elif any(x in blueprint_id for x in ["motor", "harley", "kawasaki", "yamaha", "vespa"]):
            return 3
        elif any(x in blueprint_id for x in ["bus", "volkswagen.t2"]):
            return 5
        elif any(x in blueprint_id for x in ["truck", "carlacola", "fusorosa", "cybertruck"]):
            return 7
        else:
            return 2
    return -1


def decode_depth_image(depth_image):
    """Converts the raw depth image into a 2D array of distances in meters."""
    depth = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    depth = depth.reshape((depth_image.height, depth_image.width, 4))[:, :, :3].astype(np.float32)

    r = depth[:, :, 2]
    g = depth[:, :, 1]
    b = depth[:, :, 0]

    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / ((256.0 ** 3) - 1.0)
    return 1000.0 * normalized


def decode_instance_image(instance_image):
    """
    Decodes the instance segmentation image from CARLA 0.9.15.

    CARLA stores data in BGRA byte order (OpenCV convention):
      arr[:,:,0] = B = low byte of the instance index
      arr[:,:,1] = G = high byte of the instance index
      arr[:,:,2] = R = semantic tag (e.g. 10=Vehicle, 4=Pedestrian)
      arr[:,:,3] = A = unused

    IMPORTANT: The instance index encoded here is an internal renderer index.
    It does NOT equal actor.id from the CARLA Python API. Use
    build_instance_to_actor_map() to build the pixel-id → actor mapping per frame.
    """
    arr = np.frombuffer(instance_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((instance_image.height, instance_image.width, 4))

    # R channel = semantic tag
    semantic_map = arr[:, :, 2].astype(np.uint8)

    # G<<8 | B = 16-bit internal instance index
    instance_map = (arr[:, :, 1].astype(np.uint32) << 8) | arr[:, :, 0].astype(np.uint32)

    return semantic_map, instance_map

# ==========================================
# 2. PROJECTION HELPERS
# ==========================================

def _project_world_point(point_xyz, K, w2c):
    p = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float32)
    p_camera = np.dot(w2c, p)

    if p_camera[0] <= 0.01:
        return None

    p_img = np.dot(K, np.array([p_camera[1], -p_camera[2], p_camera[0]], dtype=np.float32))
    if abs(p_img[2]) < 1e-6:
        return None

    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]

    # Reject NaN/Inf that can arise from degenerate vertex positions
    if not (np.isfinite(u) and np.isfinite(v)):
        return None

    return u, v


def get_rough_3d_bbox(actor, K, w2c, cfg):
    """
    Projects the actor's 3D bounding box onto the image plane and returns
    a 2D pixel rectangle (xmin, ymin, xmax, ymax) as a coarse search window.
    Returns None if the actor is behind the camera or outside the image.
    """
    bb = actor.bounding_box
    world_vertices = bb.get_world_vertices(actor.get_transform())

    pts = []
    for vertex in world_vertices:
        p = _project_world_point((vertex.x, vertex.y, vertex.z), K, w2c)
        if p is not None:
            pts.append(p)

    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    if not (all(np.isfinite(x) for x in xs) and all(np.isfinite(y) for y in ys)):
        return None

    xmin = max(0, int(np.floor(min(xs))))
    xmax = min(cfg.capture.img_width - 1, int(np.ceil(max(xs))))
    ymin = max(0, int(np.floor(min(ys))))
    ymax = min(cfg.capture.img_height - 1, int(np.ceil(max(ys))))

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax

# ==========================================
# 3. INSTANCE-ID → ACTOR MAPPING  
# ==========================================

def build_instance_to_actor_map(actors, ego_vehicle, K, w2c, depth_map, semantic_map, instance_map, cfg):
    """
    Builds a per-frame mapping of {instance_pixel_id: actor} for every visible actor.
    """
    mapping = {} 

    max_render_distance = getattr(cfg.capture, 'max_render_distance', 100.0)

    for actor in actors:
        if ego_vehicle is not None and actor.id == ego_vehicle.id:
            continue

        actor_loc  = actor.get_transform().location
        ego_loc    = ego_vehicle.get_transform().location
        dist_to_actor = actor_loc.distance(ego_loc)
        if dist_to_actor > max_render_distance:
            continue

        rough_bbox = get_rough_3d_bbox(actor, K, w2c, cfg)
        if rough_bbox is None:
            continue

        xmin, ymin, xmax, ymax = rough_bbox
        target_tags = TAG_VEHICLES if actor.type_id.startswith("vehicle.") else TAG_PEDESTRIANS

        sem_crop  = semantic_map[ymin:ymax + 1, xmin:xmax + 1]
        inst_crop = instance_map[ymin:ymax + 1, xmin:xmax + 1]
        dep_crop  = depth_map[ymin:ymax + 1, xmin:xmax + 1]

        ext = actor.bounding_box.extent
        bb_half_diag = float(np.sqrt(ext.x ** 2 + ext.y ** 2 + ext.z ** 2))
        tolerance = bb_half_diag + 2.0 

        valid_mask = np.isin(sem_crop, target_tags) & (np.abs(dep_crop - dist_to_actor) <= tolerance)
        candidate_ids = inst_crop[valid_mask]

        if candidate_ids.size == 0:
            continue

        counts = np.bincount(candidate_ids.astype(np.int64))
        dominant_id = int(np.argmax(counts))

        if dominant_id <= 0:
            continue

        if dominant_id not in mapping:
            mapping[dominant_id] = actor

    return mapping  

# ==========================================
# 4. PIXEL-PERFECT BOUNDING BOX 
# ==========================================

def get_pixel_perfect_bbox(inst_id, target_tags, semantic_map, instance_map, cfg):
    """
    Given a confirmed instance pixel ID, calculates a tight axis-aligned bounding box.
    It includes an occlusion check to discard boxes that are mostly empty space.
    """
    final_mask = (instance_map == inst_id) & np.isin(semantic_map, target_tags)
    
    # 1. Count how many actual pixels of this object are visible on screen
    visible_pixel_count = np.count_nonzero(final_mask)
    
    # If fewer than 20 pixels are visible, the object is virtually completely occluded.
    if visible_pixel_count < 20:
        return None

    y_coords, x_coords = np.where(final_mask)

    tight_xmin = int(np.min(x_coords))
    tight_xmax = int(np.max(x_coords))
    tight_ymin = int(np.min(y_coords))
    tight_ymax = int(np.max(y_coords))

    # 2. Calculate the total 2D area of the bounding box
    area = (tight_xmax - tight_xmin + 1) * (tight_ymax - tight_ymin + 1)
    
    if area < cfg.capture.min_box_area:
        return None

    # 3. THE OCCLUSION CHECK (Fill Ratio)
    # Divide the actual visible pixels by the total box area. 
    # If the box is less than 15% full, it means the object is heavily occluded 
    # (like behind a fence or trees). We drop it so we don't train on background data.
    fill_ratio = visible_pixel_count / float(area)
    
    if fill_ratio < 0.15:
        return None

    return tight_xmin, tight_ymin, tight_xmax, tight_ymax

# ==========================================
# 5. SPAWNING EGO VEHICLE
# ==========================================
# Note: Removed the smooth_bbox function entirely to prevent bounding box lag

def spawn_ego(world, cfg, ego_seed=None):
    seeds = resolve_seeds(cfg)
    if ego_seed is None:
        ego_seed = seeds["ego_seed"]

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on this map.")

    ego_bp = blueprint_library.find(cfg.ego_blueprint)
    ego_bp.set_attribute("role_name", cfg.ego_role_name)

    ego_rng = random.Random(ego_seed)
    ego_transform = ego_rng.choice(spawn_points)
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
    if ego_vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle. Restart CARLA and try again.")
    ego_vehicle.set_autopilot(cfg.ego_autopilot)
    return ego_vehicle

# ==========================================
# 6. MAIN CAPTURE LOOP
# ==========================================

def run_capture(world, cfg, ego_vehicle=None, ego_seed=None):
    ensure_directories(cfg)

    instance_dir = os.path.join(cfg.capture.output_dir, "instance_seg")
    os.makedirs(instance_dir, exist_ok=True)
    semantic_dir = os.path.join(cfg.capture.output_dir, "semantic_debug")
    os.makedirs(semantic_dir, exist_ok=True)

    seeds = resolve_seeds(cfg)
    if ego_seed is None:
        ego_seed = seeds["ego_seed"]

    spawned_actors = []

    try:
        if ego_vehicle is None:
            ego_vehicle = spawn_ego(world, cfg, ego_seed)
            spawned_actors.append(ego_vehicle)

        cam_transform     = carla.Transform(carla.Location(x=cfg.ego_camera_x, y=cfg.ego_camera_y, z=cfg.ego_camera_z))
        blueprint_library = world.get_blueprint_library()

        # --- Spawn cameras ---
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(cfg.capture.img_width))
        rgb_bp.set_attribute("image_size_y", str(cfg.capture.img_height))
        rgb_bp.set_attribute("fov", str(cfg.capture.fov))

        depth_bp = blueprint_library.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(cfg.capture.img_width))
        depth_bp.set_attribute("image_size_y", str(cfg.capture.img_height))
        depth_bp.set_attribute("fov", str(cfg.capture.fov))

        inst_bp = blueprint_library.find("sensor.camera.instance_segmentation")
        inst_bp.set_attribute("image_size_x", str(cfg.capture.img_width))
        inst_bp.set_attribute("image_size_y", str(cfg.capture.img_height))
        inst_bp.set_attribute("fov", str(cfg.capture.fov))

        rgb_camera   = world.spawn_actor(rgb_bp,   cam_transform, attach_to=ego_vehicle)
        depth_camera = world.spawn_actor(depth_bp,  cam_transform, attach_to=ego_vehicle)
        inst_camera  = world.spawn_actor(inst_bp,   cam_transform, attach_to=ego_vehicle)

        spawned_actors.extend([rgb_camera, depth_camera, inst_camera])

        rgb_queue   = queue.Queue()
        depth_queue = queue.Queue()
        inst_queue  = queue.Queue()

        rgb_camera.listen(rgb_queue.put)
        depth_camera.listen(depth_queue.put)
        inst_camera.listen(inst_queue.put)

        K = build_projection_matrix(cfg.capture.img_width, cfg.capture.img_height, cfg.capture.fov)
        print(f"Starting capture of {cfg.capture.max_frames} frames...")

        for frame in range(cfg.capture.max_frames):
            world.tick()

            rgb_image   = rgb_queue.get(timeout=2.0)
            depth_image = depth_queue.get(timeout=2.0)
            inst_image  = inst_queue.get(timeout=2.0)

            rgb_array  = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
            rgb_array  = rgb_array.reshape((cfg.capture.img_height, cfg.capture.img_width, 4))
            bgr_image  = rgb_array[:, :, :3].copy()
            draw_image = bgr_image.copy()

            depth_map                  = decode_depth_image(depth_image)
            semantic_map, instance_map = decode_instance_image(inst_image)
            w2c                        = np.array(rgb_camera.get_transform().get_inverse_matrix())

            vehicles = list(world.get_actors().filter("*vehicle*"))
            walkers  = list(world.get_actors().filter("*walker*"))
            actors   = vehicles + walkers

            inst_to_actor = build_instance_to_actor_map(
                actors, ego_vehicle, K, w2c,
                depth_map, semantic_map, instance_map, cfg
            )

            labels_content = ""
            detected       = 0

            for inst_id, actor in inst_to_actor.items():
                coco_class = get_coco_class(actor.type_id)
                if coco_class == -1:
                    continue

                target_tags = TAG_VEHICLES if actor.type_id.startswith("vehicle.") else TAG_PEDESTRIANS
                
                # Fetch exact bound from this current frame mask
                raw_bbox = get_pixel_perfect_bbox(inst_id, target_tags, semantic_map, instance_map, cfg)

                # If the box is dropped (due to occlusion or too few pixels), we skip processing.
                # Do NOT use a previous frame's box as it creates floating, lagging artifacts.
                if raw_bbox is None:
                    continue
                    
                bbox = raw_bbox 

                xmin, ymin, xmax, ymax = bbox

                center_x = (xmin + xmax) / 2.0 / cfg.capture.img_width
                center_y = (ymin + ymax) / 2.0 / cfg.capture.img_height
                width    = (xmax - xmin)        / cfg.capture.img_width
                height   = (ymax - ymin)        / cfg.capture.img_height

                labels_content += "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    coco_class, center_x, center_y, width, height
                )

                cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                if cfg.capture.draw_labels:
                    cv2.putText(
                        draw_image, str(coco_class),
                        (xmin, max(0, ymin - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

                detected += 1

            # --- Save outputs ---
            file_prefix = "{:05d}".format(frame)

            cv2.imwrite(os.path.join(cfg.capture.rgb_dir,  file_prefix + ".jpg"), bgr_image)
            cv2.imwrite(os.path.join(cfg.capture.bbox_dir, file_prefix + ".jpg"), draw_image)

            inst_raw_array = (
                np.frombuffer(inst_image.raw_data, dtype=np.uint8)
                  .reshape((cfg.capture.img_height, cfg.capture.img_width, 4))
            )
            cv2.imwrite(os.path.join(instance_dir, file_prefix + ".png"), inst_raw_array[:, :, :3])

            cv2.imwrite(
                os.path.join(semantic_dir, file_prefix + ".png"),
                (semantic_map * 10).astype(np.uint8)  
            )

            with open(os.path.join(cfg.capture.labels_dir, file_prefix + ".txt"), "w") as f:
                f.write(labels_content)

            print(f"Processed frame {frame + 1}/{cfg.capture.max_frames} - Detected {detected} objects")

    finally:
        for actor in reversed(spawned_actors):
            try:
                actor.destroy()
            except Exception:
                pass
        print("Done. Cleaned up actors.")

# ==========================================
# 7. ENTRY POINT
# ==========================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Capture CARLA bounding-box training data.")
    parser.add_argument("--host",        default="127.0.0.1")
    parser.add_argument("--port",        type=int,   default=2000)
    parser.add_argument("--frames",      type=int,   default=None)
    parser.add_argument("--width",       type=int,   default=1920)
    parser.add_argument("--height",      type=int,   default=1080)
    parser.add_argument("--fov",         type=float, default=90.0)
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--ego-seed",    type=int,   default=None)
    parser.add_argument("--draw-labels", action="store_true")
    args = parser.parse_args()

    cfg = SimulationConfig()
    cfg.host = args.host
    cfg.port = args.port
    if args.frames is not None:
        cfg.capture.max_frames = args.frames
    cfg.capture.img_width  = args.width
    cfg.capture.img_height = args.height
    cfg.capture.fov        = args.fov
    cfg.capture.draw_labels = args.draw_labels
    if args.seed is not None:
        cfg.master_seed = args.seed
    if args.ego_seed is not None:
        cfg.ego_seed = args.ego_seed

    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(10.0)
    world = client.get_world()

    original_settings = world.get_settings()
    try:
        from carla_utils import apply_world_settings, restore_world_settings, set_weather

        set_weather(world, cfg.weather)
        apply_world_settings(world, cfg)
        run_capture(world, cfg, ego_vehicle=None)
    finally:
        restore_world_settings(world, original_settings)


if __name__ == "__main__":
    main()