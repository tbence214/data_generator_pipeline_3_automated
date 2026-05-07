import argparse

import carla

from carla_config import SimulationConfig
# 1. Added load_world to imports
from carla_utils import apply_world_settings, restore_world_settings, set_weather, load_world
from capture_dataset import run_capture, spawn_ego
from traffic_utils import cleanup_traffic, spawn_traffic


def main():
    parser = argparse.ArgumentParser(description="Run traffic + capture in one process.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--traffic-seed", type=int, default=None)
    parser.add_argument("--ego-seed", type=int, default=None)
    parser.add_argument("--walker-seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--vehicles", type=int, default=30)
    parser.add_argument("--walkers", type=int, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--draw-labels", action="store_true")
    parser.add_argument("--car-lights-on", action="store_true")
    parser.add_argument("--safe", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    # 2. Added the map argument
    parser.add_argument("--map", type=str, default=None, help="Map to load (e.g., Town01, Town03)")
    args = parser.parse_args()

    cfg = SimulationConfig()
    cfg.host = args.host
    cfg.port = args.port
    if args.seed is not None: cfg.master_seed = args.seed
    if args.traffic_seed is not None: cfg.traffic_seed = args.traffic_seed
    if args.ego_seed is not None: cfg.ego_seed = args.ego_seed
    if args.walker_seed is not None: cfg.walker_seed = args.walker_seed
    if args.frames is not None:
        cfg.capture.max_frames = args.frames
    cfg.capture.img_width = args.width
    cfg.capture.img_height = args.height
    cfg.capture.fov = args.fov
    cfg.capture.draw_labels = args.draw_labels
    cfg.traffic.number_of_vehicles = args.vehicles
    cfg.traffic.number_of_walkers = args.walkers
    cfg.traffic.car_lights_on = args.car_lights_on
    cfg.traffic.safe = args.safe
    cfg.traffic.hybrid = args.hybrid
    # 3. Only override the config map if a command-line argument was provided
    if args.map is not None:
        cfg.map_name = args.map
        
    # UPDATE: Dynamically set the output directories based on the map name
    base_folder = f"dataset_{cfg.map_name}" if cfg.map_name else "dataset_default"
    cfg.capture.output_dir = base_folder
    cfg.capture.labels_dir = f"{base_folder}/labels"
    cfg.capture.rgb_dir = f"{base_folder}/pictures"
    cfg.capture.bbox_dir = f"{base_folder}/b_picture"

    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(10.0)
    
    # 4. Use the load_world utility to change maps via the Python API
    world = load_world(client, cfg.map_name)

    state = None
    original_settings = world.get_settings()
    ego_vehicle = None
    
    # 5. CRITICAL FIX: Initialize static_parked_state before the try block
    static_parked_state = None 
    
    try:
        set_weather(world, cfg.weather)
        # from traffic_utils import hide_static_parked_cars
        # hide_static_parked_cars(world)
        
        # 1. Import your new functions
        from traffic_utils import replace_static_parked_cars_with_actors
        
        # 2. Run the replacement (No need to call hide_static_parked_cars anymore, 
        #    because this new function hides them automatically)
        static_parked_state = replace_static_parked_cars_with_actors(client, world)
        
        ego_vehicle = spawn_ego(world, cfg, cfg.ego_seed)
        state = spawn_traffic(client, world, cfg, reserved_spawn_points=[ego_vehicle.get_transform()])

        # ==========================================
        # 3. NEW: DROP AND FREEZE DELAY
        # Let the simulation run for 5 frames to let the parked cars hit the ground
        # ==========================================
        print("Letting parked vehicles drop to the ground...")
        for _ in range(5):
            world.tick()
            
        # Now that they have landed, freeze them before the bikes tip over!
        if static_parked_state is not None:
            parked_actors = world.get_actors(static_parked_state)
            for actor in parked_actors:
                if actor is not None and actor.is_alive:
                    actor.set_simulate_physics(False)
        # ==========================================

        run_capture(world, cfg, ego_vehicle=ego_vehicle)
    finally:
        if state is not None:
            cleanup_traffic(client, world, state)
        else:
            restore_world_settings(world, original_settings)
            
        if static_parked_state is not None:
            from traffic_utils import cleanup_static_parked_replacements
            cleanup_static_parked_replacements(client, world, static_parked_state)
            
        if ego_vehicle is not None:
            try:
                ego_vehicle.destroy()
            except Exception:
                pass

if __name__ == "__main__":
    main()
