#!/usr/bin/env python
import argparse
import carla

from carla_config import SimulationConfig
from carla_utils import set_weather
from capture_dataset import spawn_ego
from traffic_utils import (
    cleanup_traffic, 
    spawn_traffic, 
    replace_static_parked_cars_with_actors, 
    cleanup_static_parked_replacements
)

def main():
    parser = argparse.ArgumentParser(description="Generate CARLA traffic with controllable seeds.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None, help="Master seed")
    parser.add_argument("--traffic-seed", type=int, default=None)
    parser.add_argument("--walker-seed", type=int, default=None)
    parser.add_argument("--vehicles", type=int, default=30)
    parser.add_argument("--walkers", type=int, default=10)
    parser.add_argument("--asynch", action="store_true", help="Do not use this if you want accurate bounding boxes!")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--car-lights-on", action="store_true")
    parser.add_argument("--respawn", action="store_true")
    parser.add_argument("--safe", action="store_true")
    args = parser.parse_args()

    cfg = SimulationConfig()
    cfg.host = args.host
    cfg.port = args.port
    if args.seed is not None: cfg.master_seed = args.seed
    if args.traffic_seed is not None: cfg.traffic_seed = args.traffic_seed
    if args.walker_seed is not None: cfg.walker_seed = args.walker_seed
    cfg.traffic.number_of_vehicles = args.vehicles
    cfg.traffic.number_of_walkers = args.walkers
    cfg.traffic.asynch = args.asynch
    cfg.traffic.hybrid = args.hybrid
    cfg.traffic.car_lights_on = args.car_lights_on
    cfg.traffic.respawn = args.respawn
    cfg.traffic.safe = args.safe

    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(10.0)
    world = client.get_world()

    state = None
    static_parked_state = None
    ego_vehicle = None
    original_settings = world.get_settings()
    
    try:
        set_weather(world, cfg.weather)
        
        # 1. Hide fake environment vehicles and place real ones
        static_parked_state = replace_static_parked_cars_with_actors(client, world)
        
        # 2. Spawn Ego Vehicle so the spawn sequence perfectly matches run_dataset.py
        ego_vehicle = spawn_ego(world, cfg)
        
        # 3. Spawn Traffic, reserving the ego vehicle's exact spot
        state = spawn_traffic(client, world, cfg, reserved_spawn_points=[ego_vehicle.get_transform()])
        
        print("Traffic generated perfectly synced with the main dataset. Press Ctrl+C to stop.")
        while True:
            # Running asynchronously causes distance tracking and bounding boxes to break. 
            # Ensure --asynch is NOT passed in the terminal.
            if not cfg.traffic.asynch and state.synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()
                
    except KeyboardInterrupt:
        pass
    finally:
        if state is not None:
            cleanup_traffic(client, world, state)
        else:
            world.apply_settings(original_settings)
            
        if static_parked_state is not None:
            cleanup_static_parked_replacements(client, world, static_parked_state)
            
        if ego_vehicle is not None:
            try:
                ego_vehicle.destroy()
            except Exception:
                pass
            
        print("done.")

if __name__ == "__main__":
    main()