"""
Frame-by-frame image capture and YOLO bounding-box label generation for CARLA.

Each simulation tick, three synchronized cameras are read:
  - RGB          → saved as the training image
  - Depth        → used to disambiguate overlapping actors by distance
  - Instance seg → used to find per-pixel actor membership

Detected actors are mapped to COCO class IDs and written as YOLO-format labels.

Reliability features:
  - The depth gate compares planar (camera-forward) depths on both sides.
    CARLA's depth camera encodes z-buffer depth, not Euclidean distance, so
    expected actor depth is taken as the forward component of the bbox centre
    in camera space. The previous Euclidean comparison rejected visible actors
    near the left/right image edges, where the two conventions diverge by up
    to √2 at a 90° FOV.
  - Sensor images are validated against the simulation frame ID so RGB, depth
    and instance buffers are guaranteed to belong to the same tick. Desynced
    frames are skipped instead of being written with misaligned labels.
  - Instance-ID ownership is resolved globally per frame (highest vote count
    wins), so a partially occluded actor can no longer "steal" the instance ID
    of the actor occluding it — both receive their correct boxes.
  - The fill-ratio occlusion filter scales with box area, so small or
    fragmented silhouettes (e.g. a pedestrian emerging from behind a car) are
    not rejected by a threshold tuned for large near-field actors.
"""
import os
import queue
import random

import carla
import cv2
import numpy as np

from carla_config import SimulationConfig
from carla_utils import (
    apply_world_settings,
    build_projection_matrix,
    ensure_directories,
    resolve_seeds,
    restore_world_settings,
    set_weather,
)

# Semantic tag constants for CARLA 0.9.15 (CityScapes label scheme).
# Vehicles covers cars, trucks, buses, motorcycles, and bicycles (tags 14–19).
# Pedestrians covers walkers and riders (tags 12–13).
TAG_VEHICLES    = (14, 15, 16, 18, 19)
TAG_PEDESTRIANS = (12, 13)

# Padding added to every side of the rough 3-D bbox crop window so that actors
# sitting at the image edge still get enough pixels for reliable instance-ID voting.
_CROP_PAD = 20

# Minimum number of pixels an actor must accumulate for an instance ID before
# that (actor, instance ID) pair is considered a valid candidate. Prevents a
# handful of stray pixels from another actor deciding the vote.
_MIN_VOTE_PIXELS = 8

# How many stale sensor images to drain per frame before giving up.
_MAX_STALE_IMAGES = 5


# ──────────────────────────────────────────────────────────────
# 1. UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────

def get_coco_class(blueprint_id):
    """Maps a CARLA blueprint ID to its COCO category index."""
    if blueprint_id.startswith("walker."):
        return 0

    if blueprint_id.startswith("vehicle."):
        if any(x in blueprint_id for x in ["bicycle", "crossbike", "century", "omafiets"]):
            return 1
        elif any(x in blueprint_id for x in ["motorcycle", "harley", "kawasaki", "yamaha", "vespa"]):
            return 3
        elif any(x in blueprint_id for x in ["bus", "fusorosa"]):
            return 5
        elif any(x in blueprint_id for x in [
            "truck", "carlacola", "european_hgv", "firetruck", "cybertruck",
            "van", "ambulance", "sprinter", "volkswagen.t2"
        ]):
            return 7
        else:
            return 2

    return -1


def decode_depth_image(depth_image):
    """Converts the raw CARLA depth buffer to a 2-D array of distances in metres."""
    depth = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    depth = depth.reshape((depth_image.height, depth_image.width, 4))[:, :, :3].astype(np.float32)

    r = depth[:, :, 2]
    g = depth[:, :, 1]
    b = depth[:, :, 0]

    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / ((256.0 ** 3) - 1.0)
    return 1000.0 * normalized


def decode_instance_image(instance_image):
    """
    Decodes the CARLA instance segmentation image into semantic and instance maps.

    CARLA stores pixels in BGRA byte order (OpenCV convention):
      channel 0 (B) = low byte of the internal instance index
      channel 1 (G) = high byte of the internal instance index
      channel 2 (R) = semantic tag (e.g. Vehicle, Pedestrian)
      channel 3 (A) = unused

    The instance index is an internal renderer index and does NOT equal actor.id
    from the Python API. Use build_instance_to_actor_map() to build the
    pixel-id → actor mapping per frame.
    """
    arr = np.frombuffer(instance_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((instance_image.height, instance_image.width, 4))

    semantic_map = arr[:, :, 2].astype(np.uint8)
    instance_map = (arr[:, :, 1].astype(np.uint32) << 8) | arr[:, :, 0].astype(np.uint32)

    return semantic_map, instance_map


# ──────────────────────────────────────────────────────────────
# 2. SENSOR FRAME SYNCHRONIZATION
# ──────────────────────────────────────────────────────────────

class _SensorBuffer:
    """
    Wraps a sensor queue and guarantees that get() returns the image belonging
    to a specific simulation frame.

    Stale images (from earlier ticks, e.g. duplicates produced right after the
    sensor spawns) are silently dropped. Images that arrive *early* (a future
    frame) are stashed and served later, so a single hiccup never causes a
    permanent off-by-one between the RGB, depth and instance streams.
    """

    def __init__(self, name):
        self.name = name
        self.queue = queue.Queue()
        self._stash = {}

    def put(self, image):
        self.queue.put(image)

    def get(self, expected_frame, timeout=2.0):
        # Prune stash entries that can never be requested again.
        for f in [f for f in self._stash if f < expected_frame]:
            del self._stash[f]

        if expected_frame in self._stash:
            return self._stash.pop(expected_frame)

        for _ in range(_MAX_STALE_IMAGES):
            image = self.queue.get(timeout=timeout)
            if image.frame == expected_frame:
                return image
            if image.frame > expected_frame:
                # Arrived early — keep it for the next tick, report a skip now.
                self._stash[image.frame] = image
                raise LookupError(
                    f"{self.name} sensor skipped frame {expected_frame} "
                    f"(next available: {image.frame})"
                )
            # image.frame < expected_frame → stale duplicate, drop and retry.

        raise LookupError(
            f"{self.name} sensor produced {_MAX_STALE_IMAGES} stale images "
            f"while waiting for frame {expected_frame}"
        )


# ──────────────────────────────────────────────────────────────
# 3. PROJECTION HELPERS
# ──────────────────────────────────────────────────────────────

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

    if not (np.isfinite(u) and np.isfinite(v)):
        return None

    return u, v


def get_rough_3d_bbox(actor, K, w2c, cfg):
    """
    Projects the actor's 3-D bounding box onto the image plane and returns a
    padded 2-D pixel rectangle (xmin, ymin, xmax, ymax) as a search window.

    If all 8 corner vertices are behind the camera (e.g. an actor very close to
    the lens), the function falls back to projecting the bounding-box centre so
    that partially-visible actors near the near-clip plane are not silently dropped.
    Returns None only when the entire projected region lies outside the image.
    """
    bb = actor.bounding_box
    world_vertices = bb.get_world_vertices(actor.get_transform())

    pts = []
    for vertex in world_vertices:
        p = _project_world_point((vertex.x, vertex.y, vertex.z), K, w2c)
        if p is not None:
            pts.append(p)

    # All corners behind the camera — fall back to projecting the bbox centre.
    if not pts:
        bb_center_world = actor.get_transform().transform(actor.bounding_box.location)
        p = _project_world_point(
            (bb_center_world.x, bb_center_world.y, bb_center_world.z), K, w2c
        )
        if p is None:
            return None
        pts = [p]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    if not (all(np.isfinite(x) for x in xs) and all(np.isfinite(y) for y in ys)):
        return None

    raw_xmin = int(np.floor(min(xs)))
    raw_xmax = int(np.ceil(max(xs)))
    raw_ymin = int(np.floor(min(ys)))
    raw_ymax = int(np.ceil(max(ys)))

    if raw_xmax < 0 or raw_xmin >= cfg.capture.img_width:
        return None
    if raw_ymax < 0 or raw_ymin >= cfg.capture.img_height:
        return None

    xmin = max(0,                          raw_xmin - _CROP_PAD)
    xmax = min(cfg.capture.img_width  - 1, raw_xmax + _CROP_PAD)
    ymin = max(0,                          raw_ymin - _CROP_PAD)
    ymax = min(cfg.capture.img_height - 1, raw_ymax + _CROP_PAD)

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax


# ──────────────────────────────────────────────────────────────
# 4. INSTANCE-ID → ACTOR MAPPING
# ──────────────────────────────────────────────────────────────

def _get_target_tags(actor):
    """Returns the semantic tags an actor's pixels may carry."""
    coco_class = get_coco_class(actor.type_id)
    # Two-wheeled vehicles share semantic pixels with their rider.
    if coco_class in (1, 3):
        return TAG_VEHICLES + TAG_PEDESTRIANS
    return TAG_VEHICLES if actor.type_id.startswith("vehicle.") else TAG_PEDESTRIANS


def build_instance_to_actor_map(
    actors, ego_vehicle, cam_loc, K, w2c,
    depth_map, semantic_map, instance_map, cfg
):
    """
    Builds a per-frame {instance_pixel_id: actor} mapping for every visible actor.

    Phase 1 — candidate collection. For each actor:
      1. Cull by distance from the camera lens.
      2. Project the 3-D bounding box to get a pixel search window.
      3. Filter pixels by semantic tag and depth.
      4. Record a vote (pixel count) for EVERY instance ID found in the window,
         not just the dominant one.

    Phase 2 — global conflict resolution. All (votes, instance ID, actor)
    candidates are sorted by vote count and assigned greedily, with each
    instance ID and each actor used at most once. This way an actor that is
    partially occluded by another actor of the same semantic class no longer
    claims the occluder's instance ID (the occluder always has more votes for
    its own ID), and instead falls through to claiming its own smaller blob.
    Previously such conflicts silently dropped one of the two actors and could
    attach the wrong class label to the survivor.

    IMPORTANT — depth convention. The CARLA depth camera encodes PLANAR depth:
    the distance along the camera's forward axis (z-buffer depth), NOT the
    Euclidean distance from the lens. The expected actor depth must therefore
    be the forward component of the bounding-box centre in camera coordinates,
    obtained via the world-to-camera matrix. Comparing the depth map against a
    Euclidean distance instead introduces an error that grows with the angle
    from the optical axis — at a 90° FOV it reaches a factor of √2 at the image
    edge, which silently rejected perfectly visible actors near the left and
    right borders of the frame. The bounding-box centre (rather than the
    ground-level root pivot) is still used to avoid a vertical bias on actors
    close to the camera.
    """
    max_render_distance = getattr(cfg.capture, 'max_render_distance', 100.0)
    candidates = []  # (vote_count, instance_id, actor)

    for actor in actors:
        if ego_vehicle is not None and actor.id == ego_vehicle.id:
            continue

        actor_loc       = actor.get_transform().location
        bb_center_world = actor.get_transform().transform(actor.bounding_box.location)

        # Expected depth on the SAME scale as the depth map: forward component
        # of the bbox centre in camera space (planar depth), not Euclidean.
        p_cam = np.dot(w2c, np.array(
            [bb_center_world.x, bb_center_world.y, bb_center_world.z, 1.0],
            dtype=np.float64
        ))
        expected_depth = max(float(p_cam[0]), 0.1)

        if actor_loc.distance(cam_loc) > max_render_distance:
            continue

        rough_bbox = get_rough_3d_bbox(actor, K, w2c, cfg)
        if rough_bbox is None:
            continue

        xmin, ymin, xmax, ymax = rough_bbox
        target_tags = _get_target_tags(actor)

        sem_crop  = semantic_map[ymin:ymax + 1, xmin:xmax + 1]
        inst_crop = instance_map[ymin:ymax + 1, xmin:xmax + 1]
        dep_crop  = depth_map[ymin:ymax + 1,   xmin:xmax + 1]

        ext = actor.bounding_box.extent
        bb_half_diag = float(np.sqrt(ext.x ** 2 + ext.y ** 2 + ext.z ** 2))
        tolerance = bb_half_diag + cfg.capture.depth_tolerance_meters

        valid_mask    = np.isin(sem_crop, target_tags) & (np.abs(dep_crop - expected_depth) <= tolerance)
        candidate_ids = inst_crop[valid_mask]

        # Retry with a wider depth window for fast-moving or steeply-angled actors
        # where the bbox-centre depth diverges slightly from the visible surface depth.
        if candidate_ids.size == 0:
            fallback_mask = np.isin(sem_crop, target_tags) & (
                np.abs(dep_crop - expected_depth) <= tolerance * 2.0
            )
            candidate_ids = inst_crop[fallback_mask]

        if candidate_ids.size == 0:
            continue

        unique_ids, counts = np.unique(candidate_ids, return_counts=True)
        for inst_id, votes in zip(unique_ids, counts):
            if inst_id > 0 and votes >= _MIN_VOTE_PIXELS:
                candidates.append((int(votes), int(inst_id), actor))

    # Highest vote counts claim their instance ID first; each actor and each
    # instance ID can only be assigned once.
    candidates.sort(key=lambda c: c[0], reverse=True)

    mapping = {}
    claimed_actor_ids = set()
    for votes, inst_id, actor in candidates:
        if inst_id in mapping or actor.id in claimed_actor_ids:
            continue
        mapping[inst_id] = actor
        claimed_actor_ids.add(actor.id)

    return mapping


# ──────────────────────────────────────────────────────────────
# 5. PIXEL-PERFECT BOUNDING BOX
# ──────────────────────────────────────────────────────────────

def _min_fill_ratio_for_area(area, cfg):
    """
    Returns the minimum acceptable fill ratio for a tight box of the given area.

    Small or fragmented silhouettes (a pedestrian half-emerged from behind a
    car shows only head and feet, with most of the tight box empty) need a
    lenient threshold, while large near-field actors keep a stricter one so
    ghost boxes through fences and foliage are still rejected. The configured
    min_visible_ratio acts as an upper bound on the strict end.
    """
    strict  = min(0.10, cfg.capture.min_visible_ratio)
    lenient = min(0.05, strict)
    return float(np.interp(area, [400.0, 20000.0], [lenient, strict]))


def get_pixel_perfect_bbox(inst_id, target_tags, semantic_map, instance_map, cfg):
    """
    Returns a tight axis-aligned bounding box for a confirmed instance pixel ID.

    Applies two quality filters before returning:
      - Minimum visible pixel count (rejects nearly-fully-occluded actors).
        This runs FIRST so the lenient fill ratio below can never admit an
        actor that is essentially invisible (e.g. behind a solid wall).
      - Area-scaled fill ratio (rejects actors behind fences or foliage where
        the tight box is mostly background pixels rather than the actor itself,
        without discarding small or partially occluded silhouettes).
    """
    final_mask          = (instance_map == inst_id) & np.isin(semantic_map, target_tags)
    visible_pixel_count = np.count_nonzero(final_mask)

    if visible_pixel_count < 12:
        return None

    y_coords, x_coords = np.where(final_mask)

    tight_xmin = int(np.min(x_coords))
    tight_xmax = int(np.max(x_coords))
    tight_ymin = int(np.min(y_coords))
    tight_ymax = int(np.max(y_coords))

    area = (tight_xmax - tight_xmin + 1) * (tight_ymax - tight_ymin + 1)
    if area < cfg.capture.min_box_area:
        return None

    fill_ratio = visible_pixel_count / float(area)
    if fill_ratio < _min_fill_ratio_for_area(area, cfg):
        return None

    return tight_xmin, tight_ymin, tight_xmax, tight_ymax


# ──────────────────────────────────────────────────────────────
# 6. EGO VEHICLE SPAWNING
# ──────────────────────────────────────────────────────────────

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

    ego_rng       = random.Random(ego_seed)
    ego_transform = ego_rng.choice(spawn_points)
    ego_vehicle   = world.try_spawn_actor(ego_bp, ego_transform)
    if ego_vehicle is None:
        raise RuntimeError("Could not spawn ego vehicle. Restart CARLA and try again.")
    ego_vehicle.set_autopilot(cfg.ego_autopilot)
    return ego_vehicle


# ──────────────────────────────────────────────────────────────
# 7. MAIN CAPTURE LOOP
# ──────────────────────────────────────────────────────────────

def run_capture(world, cfg, ego_vehicle=None, ego_seed=None):
    ensure_directories(cfg)

    instance_dir = os.path.join(cfg.capture.output_dir, "instance_seg")
    semantic_dir = os.path.join(cfg.capture.output_dir, "semantic_debug")
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)

    seeds = resolve_seeds(cfg)
    if ego_seed is None:
        ego_seed = seeds["ego_seed"]

    spawned_actors = []

    try:
        if ego_vehicle is None:
            ego_vehicle = spawn_ego(world, cfg, ego_seed)
            spawned_actors.append(ego_vehicle)

        cam_transform     = carla.Transform(carla.Location(
            x=cfg.ego_camera_x, y=cfg.ego_camera_y, z=cfg.ego_camera_z
        ))
        blueprint_library = world.get_blueprint_library()

        def _configure_camera_bp(sensor_type):
            bp = blueprint_library.find(sensor_type)
            bp.set_attribute("image_size_x", str(cfg.capture.img_width))
            bp.set_attribute("image_size_y", str(cfg.capture.img_height))
            bp.set_attribute("fov",          str(cfg.capture.fov))
            return bp

        rgb_camera   = world.spawn_actor(_configure_camera_bp("sensor.camera.rgb"),
                                         cam_transform, attach_to=ego_vehicle)
        depth_camera = world.spawn_actor(_configure_camera_bp("sensor.camera.depth"),
                                         cam_transform, attach_to=ego_vehicle)
        inst_camera  = world.spawn_actor(_configure_camera_bp("sensor.camera.instance_segmentation"),
                                         cam_transform, attach_to=ego_vehicle)
        spawned_actors.extend([rgb_camera, depth_camera, inst_camera])

        rgb_buffer   = _SensorBuffer("rgb")
        depth_buffer = _SensorBuffer("depth")
        inst_buffer  = _SensorBuffer("instance")

        rgb_camera.listen(rgb_buffer.put)
        depth_camera.listen(depth_buffer.put)
        inst_camera.listen(inst_buffer.put)

        K = build_projection_matrix(cfg.capture.img_width, cfg.capture.img_height, cfg.capture.fov)
        print(f"Starting capture of {cfg.capture.max_frames} frames...")

        skipped_frames = 0

        for frame in range(cfg.capture.max_frames):
            tick_frame = world.tick()

            # All three images must belong to this exact simulation frame,
            # otherwise labels would be computed from a stale instance/depth
            # buffer and drawn onto the wrong RGB image.
            try:
                rgb_image   = rgb_buffer.get(tick_frame)
                depth_image = depth_buffer.get(tick_frame)
                inst_image  = inst_buffer.get(tick_frame)
            except (queue.Empty, LookupError) as exc:
                skipped_frames += 1
                print(f"Frame {frame + 1}/{cfg.capture.max_frames} — "
                      f"sensor desync, skipping ({exc})")
                continue

            rgb_array  = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
            rgb_array  = rgb_array.reshape((cfg.capture.img_height, cfg.capture.img_width, 4))
            bgr_image  = rgb_array[:, :, :3].copy()
            draw_image = bgr_image.copy()

            depth_map                  = decode_depth_image(depth_image)
            semantic_map, instance_map = decode_instance_image(inst_image)
            w2c                        = np.array(rgb_camera.get_transform().get_inverse_matrix())
            cam_loc                    = rgb_camera.get_transform().location

            vehicles = list(world.get_actors().filter("*vehicle*"))
            # Use the precise pedestrian filter to avoid matching controller.ai.walker actors.
            walkers  = list(world.get_actors().filter("walker.pedestrian.*"))
            actors   = vehicles + walkers

            inst_to_actor = build_instance_to_actor_map(
                actors, ego_vehicle, cam_loc, K, w2c,
                depth_map, semantic_map, instance_map, cfg
            )

            labels_content = ""
            detected       = 0

            for inst_id, actor in inst_to_actor.items():
                coco_class = get_coco_class(actor.type_id)
                if coco_class == -1:
                    continue

                is_two_wheeled = coco_class in (1, 3)

                if is_two_wheeled:
                    # Emit separate boxes for the vehicle frame and the rider.
                    vehicle_bbox = get_pixel_perfect_bbox(
                        inst_id, TAG_VEHICLES, semantic_map, instance_map, cfg
                    )
                    if vehicle_bbox is not None:
                        xmin, ymin, xmax, ymax = vehicle_bbox
                        cx = (xmin + xmax) / 2.0 / cfg.capture.img_width
                        cy = (ymin + ymax) / 2.0 / cfg.capture.img_height
                        w  = (xmax - xmin)        / cfg.capture.img_width
                        h  = (ymax - ymin)        / cfg.capture.img_height
                        labels_content += f"{coco_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                        cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        if cfg.capture.draw_labels:
                            cv2.putText(draw_image, str(coco_class), (xmin, max(0, ymin - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected += 1

                    rider_bbox = get_pixel_perfect_bbox(
                        inst_id, TAG_PEDESTRIANS, semantic_map, instance_map, cfg
                    )
                    if rider_bbox is not None:
                        xmin, ymin, xmax, ymax = rider_bbox
                        cx = (xmin + xmax) / 2.0 / cfg.capture.img_width
                        cy = (ymin + ymax) / 2.0 / cfg.capture.img_height
                        w  = (xmax - xmin)        / cfg.capture.img_width
                        h  = (ymax - ymin)        / cfg.capture.img_height
                        labels_content += f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                        cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        if cfg.capture.draw_labels:
                            cv2.putText(draw_image, "0", (xmin, max(0, ymin - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        detected += 1

                else:
                    target_tags = TAG_VEHICLES if actor.type_id.startswith("vehicle.") else TAG_PEDESTRIANS
                    bbox = get_pixel_perfect_bbox(inst_id, target_tags, semantic_map, instance_map, cfg)
                    if bbox is None:
                        continue
                    xmin, ymin, xmax, ymax = bbox
                    cx = (xmin + xmax) / 2.0 / cfg.capture.img_width
                    cy = (ymin + ymax) / 2.0 / cfg.capture.img_height
                    w  = (xmax - xmin)        / cfg.capture.img_width
                    h  = (ymax - ymin)        / cfg.capture.img_height
                    labels_content += f"{coco_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    if cfg.capture.draw_labels:
                        cv2.putText(draw_image, str(coco_class), (xmin, max(0, ymin - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected += 1

            file_prefix = "{:05d}".format(frame)

            cv2.imwrite(os.path.join(cfg.capture.rgb_dir,  file_prefix + ".jpg"), bgr_image)
            cv2.imwrite(os.path.join(cfg.capture.bbox_dir, file_prefix + ".jpg"), draw_image)
            cv2.imwrite(
                os.path.join(instance_dir, file_prefix + ".png"),
                np.frombuffer(inst_image.raw_data, dtype=np.uint8)
                  .reshape((cfg.capture.img_height, cfg.capture.img_width, 4))[:, :, :3]
            )
            # Multiply by 9 (not 10) and clip so tags above 25 don't wrap around
            # in uint8 and show up as near-black in the debug image.
            cv2.imwrite(
                os.path.join(semantic_dir, file_prefix + ".png"),
                np.clip(semantic_map.astype(np.uint16) * 9, 0, 255).astype(np.uint8)
            )
            with open(os.path.join(cfg.capture.labels_dir, file_prefix + ".txt"), "w") as f:
                f.write(labels_content)

            print(f"Frame {frame + 1}/{cfg.capture.max_frames} — {detected} objects detected")

        if skipped_frames:
            print(f"Skipped {skipped_frames} desynced frame(s) to protect label quality.")

    finally:
        for actor in reversed(spawned_actors):
            try:
                actor.destroy()
            except Exception:
                pass
        print("Capture complete. Sensors cleaned up.")


# ──────────────────────────────────────────────────────────────
# 8. STANDALONE ENTRY POINT
# ──────────────────────────────────────────────────────────────

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
    cfg.capture.img_width   = args.width
    cfg.capture.img_height  = args.height
    cfg.capture.fov         = args.fov
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
        set_weather(world, cfg.weather)
        apply_world_settings(world, cfg)
        run_capture(world, cfg, ego_vehicle=None)
    finally:
        restore_world_settings(world, original_settings)


if __name__ == "__main__":
    main()