import os
import queue
import random

import carla
import cv2
import numpy as np

from carla_config import SimulationConfig
from carla_utils import build_projection_matrix, ensure_directories, resolve_seeds


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
    depth = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    depth = depth.reshape((depth_image.height, depth_image.width, 4))[:, :, :3].astype(np.uint32)

    r = depth[:, :, 2]
    g = depth[:, :, 1]
    b = depth[:, :, 0]

    normalized = (r + g * 256 + b * 256 * 256).astype(np.float32) / float((256 ** 3) - 1)
    return 1000.0 * normalized


def get_actor_bbox_2d(actor, K, w2c, cfg):
    bb = actor.bounding_box
    world_vertices = bb.get_world_vertices(actor.get_transform())

    pts = []
    for vertex in world_vertices:
        p = np.array([vertex.x, vertex.y, vertex.z, 1.0])
        p_camera = np.dot(w2c, p)

        if p_camera[0] <= 0.01:
            continue

        p_img = np.dot(K, np.array([p_camera[1], -p_camera[2], p_camera[0]]))
        u = p_img[0] / p_img[2]
        v_ = p_img[1] / p_img[2]
        pts.append((u, v_))

    if not pts:
        return None

    xmin = max(0, int(min(p[0] for p in pts)))
    xmax = min(cfg.capture.img_width - 1, int(max(p[0] for p in pts)))
    ymin = max(0, int(min(p[1] for p in pts)))
    ymax = min(cfg.capture.img_height - 1, int(max(p[1] for p in pts)))

    if xmax <= xmin or ymax <= ymin:
        return None

    area = (xmax - xmin) * (ymax - ymin)
    if area < cfg.capture.min_box_area:
        return None

    return xmin, ymin, xmax, ymax


def get_actor_depth_range(actor, w2c):
    bb = actor.bounding_box
    world_vertices = bb.get_world_vertices(actor.get_transform())

    depths = []
    for vertex in world_vertices:
        p = np.array([vertex.x, vertex.y, vertex.z, 1.0])
        p_camera = np.dot(w2c, p)
        if p_camera[0] > 0.01:
            depths.append(float(p_camera[0]))

    if not depths:
        return None
    return min(depths), max(depths)


def is_bbox_visible(actor, bbox, depth_map, w2c, cfg):
    depth_range = get_actor_depth_range(actor, w2c)
    if depth_range is None:
        return False

    actor_min_depth, actor_max_depth = depth_range
    xmin, ymin, xmax, ymax = bbox
    xs = np.linspace(xmin + 2, xmax - 2, cfg.capture.visible_sample_grid, dtype=int)
    ys = np.linspace(ymin + 2, ymax - 2, cfg.capture.visible_sample_grid, dtype=int)

    visible = 0
    total = 0

    for x in xs:
        if x < 0 or x >= cfg.capture.img_width:
            continue
        for y in ys:
            if y < 0 or y >= cfg.capture.img_height:
                continue

            total += 1
            d = float(depth_map[y, x])
            if d <= 0.1 or d >= 999.0:
                continue
            if (actor_min_depth - cfg.capture.depth_tolerance_meters) <= d <= (actor_max_depth + cfg.capture.depth_tolerance_meters):
                visible += 1

    if total == 0:
        return False
    return (visible / total) >= cfg.capture.min_visible_ratio


def smooth_bbox(previous_bbox, current_bbox, alpha=0.7):
    prev = np.array(previous_bbox, dtype=np.float32)
    curr = np.array(current_bbox, dtype=np.float32)
    smoothed = alpha * curr + (1.0 - alpha) * prev
    return tuple(int(round(v)) for v in smoothed)


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


def run_capture(world, cfg, ego_vehicle=None, ego_seed=None):
    ensure_directories(cfg)

    seeds = resolve_seeds(cfg)
    if ego_seed is None:
        ego_seed = seeds["ego_seed"]

    spawned_actors = []
    rgb_camera = None
    depth_camera = None
    created_ego = False
    bbox_history = {}
    bbox_missing_counts = {}
    bbox_smoothing_alpha = 0.7
    bbox_hold_frames = 4

    try:
        if ego_vehicle is None:
            ego_vehicle = spawn_ego(world, cfg, ego_seed)
            created_ego = True
            spawned_actors.append(ego_vehicle)

        cam_transform = carla.Transform(carla.Location(x=cfg.ego_camera_x, y=cfg.ego_camera_y, z=cfg.ego_camera_z))
        blueprint_library = world.get_blueprint_library()

        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(cfg.capture.img_width))
        rgb_bp.set_attribute("image_size_y", str(cfg.capture.img_height))
        rgb_bp.set_attribute("fov", str(cfg.capture.fov))

        depth_bp = blueprint_library.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", str(cfg.capture.img_width))
        depth_bp.set_attribute("image_size_y", str(cfg.capture.img_height))
        depth_bp.set_attribute("fov", str(cfg.capture.fov))

        rgb_camera = world.spawn_actor(rgb_bp, cam_transform, attach_to=ego_vehicle)
        depth_camera = world.spawn_actor(depth_bp, cam_transform, attach_to=ego_vehicle)
        spawned_actors.extend([rgb_camera, depth_camera])

        rgb_queue = queue.Queue()
        depth_queue = queue.Queue()
        rgb_camera.listen(rgb_queue.put)
        depth_camera.listen(depth_queue.put)

        K = build_projection_matrix(cfg.capture.img_width, cfg.capture.img_height, cfg.capture.fov)
        print("Starting capture of %d frames..." % cfg.capture.max_frames)

        for frame in range(cfg.capture.max_frames):
            world.tick()

            rgb_image = rgb_queue.get(timeout=2.0)
            depth_image = depth_queue.get(timeout=2.0)

            rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
            rgb_array = rgb_array.reshape((cfg.capture.img_height, cfg.capture.img_width, 4))

            bgr_image = rgb_array[:, :, :3].copy()
            draw_image = bgr_image.copy()

            depth_map = decode_depth_image(depth_image)
            w2c = np.array(rgb_camera.get_transform().get_inverse_matrix())

            vehicles = list(world.get_actors().filter("*vehicle*"))
            walkers = list(world.get_actors().filter("*walker*"))
            actors = vehicles + walkers

            labels_content = ""
            detected = 0

            for actor in actors:
                if ego_vehicle is not None and actor.id == ego_vehicle.id:
                    continue

                coco_class = get_coco_class(actor.type_id)
                if coco_class == -1:
                    continue

                raw_bbox = get_actor_bbox_2d(actor, K, w2c, cfg)

                if raw_bbox is None or not is_bbox_visible(actor, raw_bbox, depth_map, w2c, cfg):
                    if actor.id in bbox_history and bbox_missing_counts.get(actor.id, 0) < bbox_hold_frames:
                        bbox = bbox_history[actor.id]
                        bbox_missing_counts[actor.id] = bbox_missing_counts.get(actor.id, 0) + 1
                    else:
                        bbox_history.pop(actor.id, None)
                        bbox_missing_counts.pop(actor.id, None)
                        continue
                else:
                    if actor.id in bbox_history:
                        bbox = smooth_bbox(bbox_history[actor.id], raw_bbox, bbox_smoothing_alpha)
                    else:
                        bbox = raw_bbox
                    bbox_history[actor.id] = bbox
                    bbox_missing_counts[actor.id] = 0

                xmin, ymin, xmax, ymax = bbox
                center_x = (xmin + xmax) / 2.0 / cfg.capture.img_width
                center_y = (ymin + ymax) / 2.0 / cfg.capture.img_height
                width = (xmax - xmin) / cfg.capture.img_width
                height = (ymax - ymin) / cfg.capture.img_height

                labels_content += "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(coco_class, center_x, center_y, width, height)

                cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                if cfg.capture.draw_labels:
                    cv2.putText(draw_image, str(coco_class), (xmin, max(0, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detected += 1

            file_prefix = "{:05d}".format(frame)
            cv2.imwrite(os.path.join(cfg.capture.rgb_dir, file_prefix + ".jpg"), bgr_image)
            cv2.imwrite(os.path.join(cfg.capture.bbox_dir, file_prefix + ".jpg"), draw_image)

            with open(os.path.join(cfg.capture.labels_dir, file_prefix + ".txt"), "w") as f:
                f.write(labels_content)

            print("Processed frame {}/{} - Detected {} objects".format(frame + 1, cfg.capture.max_frames, detected))

    finally:
        for actor in reversed(spawned_actors):
            try:
                actor.destroy()
            except Exception:
                pass
        print("Done. Cleaned up actors.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Capture CARLA bounding-box training data.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ego-seed", type=int, default=None)
    parser.add_argument("--draw-labels", action="store_true")
    args = parser.parse_args()

    cfg = SimulationConfig()
    cfg.host = args.host
    cfg.port = args.port
    if args.frames is not None:
        cfg.capture.max_frames = args.frames
    cfg.capture.img_width = args.width
    cfg.capture.img_height = args.height
    cfg.capture.fov = args.fov
    cfg.capture.draw_labels = args.draw_labels
    if args.seed is not None: cfg.master_seed = args.seed
    if args.ego_seed is not None: cfg.ego_seed = args.ego_seed

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