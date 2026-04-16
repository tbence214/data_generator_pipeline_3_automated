import logging
import time
from dataclasses import dataclass

import carla
import numpy as np

from carla_utils import apply_world_settings, restore_world_settings, resolve_seeds


@dataclass
class TrafficState:
    vehicles_list: list
    walkers_list: list
    all_id: list
    synchronous_master: bool
    original_settings: object


def get_actor_blueprints(world, filter_pattern, generation):
    bps = world.get_blueprint_library().filter(filter_pattern)

    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            return [x for x in bps if int(x.get_attribute('generation')) == int_generation]
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
    except Exception:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def _transform_matches(a, b, loc_tol=0.05, rot_tol=0.5):
    return (
        abs(a.location.x - b.location.x) <= loc_tol and
        abs(a.location.y - b.location.y) <= loc_tol and
        abs(a.location.z - b.location.z) <= loc_tol and
        abs(a.rotation.pitch - b.rotation.pitch) <= rot_tol and
        abs(a.rotation.yaw - b.rotation.yaw) <= rot_tol and
        abs(a.rotation.roll - b.rotation.roll) <= rot_tol
    )

# HIDE PARKED VEHICLES
def hide_static_parked_cars(world):
    """
    Finds all baked-in static environment vehicles (like fake parked cars)
    and hides them from the renderer.
    Checks multiple label names to support both old and new CARLA versions (e.g., Town10HD).
    """
    env_vehicle_ids = set()
    possible_labels = ["Vehicles", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
    for label_name in possible_labels:
        if hasattr(carla.CityObjectLabel, label_name):
            label = getattr(carla.CityObjectLabel, label_name)
            env_objects = world.get_environment_objects(label)
            for obj in env_objects:
                env_vehicle_ids.add(obj.id)
    if env_vehicle_ids:
        world.enable_environment_objects(env_vehicle_ids, False)
        print(f"Successfully hid {len(env_vehicle_ids)} static map vehicles.")
    else:
        print("No static map vehicles found to hide.")


def spawn_traffic(client, world, cfg, reserved_spawn_points=None):
    seeds = resolve_seeds(cfg)
    traffic_seed = seeds["traffic_seed"]
    walker_seed = seeds["walker_seed"]

    traffic_manager = client.get_trafficmanager(cfg.traffic.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    if cfg.traffic.respawn:
        traffic_manager.set_respawn_dormant_vehicles(True)
    if cfg.traffic.hybrid:
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
    if traffic_seed is not None:
        traffic_manager.set_random_device_seed(int(traffic_seed))

    original_settings, synchronous_master = apply_world_settings(world, cfg, traffic_manager)
    if cfg.traffic.no_rendering:
        settings = world.get_settings()
        settings.no_rendering_mode = True
        world.apply_settings(settings)

    vehicles_list = []
    walkers_list = []
    all_id = []

    blueprints = get_actor_blueprints(world, cfg.traffic.filterv, cfg.traffic.generationv)
    if not blueprints:
        raise ValueError("Couldn't find any vehicles with the specified filters")
    blueprints_walkers = get_actor_blueprints(world, cfg.traffic.filterw, cfg.traffic.generationw)
    if not blueprints_walkers:
        raise ValueError("Couldn't find any walkers with the specified filters")

    if cfg.traffic.safe:
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    if reserved_spawn_points:
        spawn_points = [
            sp for sp in spawn_points
            if not any(_transform_matches(sp, reserved) for reserved in reserved_spawn_points)
        ]
    number_of_spawn_points = len(spawn_points)

    rng = np.random.RandomState(traffic_seed)

    if cfg.traffic.number_of_vehicles < number_of_spawn_points:
        rng.shuffle(spawn_points)
    elif cfg.traffic.number_of_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, cfg.traffic.number_of_vehicles, number_of_spawn_points)
        cfg.traffic.number_of_vehicles = number_of_spawn_points

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    hero = cfg.traffic.hero
    for n, transform in enumerate(spawn_points):
        if n >= cfg.traffic.number_of_vehicles:
            break
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = rng.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = rng.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if hero:
            blueprint.set_attribute('role_name', 'hero')
            hero = False
        else:
            blueprint.set_attribute('role_name', 'autopilot')

        batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    if cfg.traffic.car_lights_on and vehicles_list:
        all_vehicle_actors = world.get_actors(vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)

    percentage_pedestrians_running = 0.0
    percentage_pedestrians_crossing = 0.0

    if walker_seed is not None:
        world.set_pedestrians_seed(int(walker_seed))
        rng = np.random.RandomState(walker_seed)

    walker_spawn_points = []
    for _ in range(cfg.traffic.number_of_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            walker_spawn_points.append(spawn_point)

    batch = []
    walker_speed = []
    for spawn_point in walker_spawn_points:
        walker_bp = rng.choice(blueprints_walkers)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        if walker_bp.has_attribute('speed'):
            if rng.random_sample() > percentage_pedestrians_running:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))

    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2

    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id

    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])

    all_actors = world.get_actors(all_id)
    if cfg.traffic.asynch or not synchronous_master:
        world.wait_for_tick()
    else:
        world.tick()

    world.set_pedestrians_cross_factor(percentage_pedestrians_crossing)
    for i in range(0, len(all_id), 2):
        all_actors[i].start()
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))
    traffic_manager.global_percentage_speed_difference(30.0)

    return TrafficState(
        vehicles_list=vehicles_list,
        walkers_list=walkers_list,
        all_id=all_id,
        synchronous_master=synchronous_master,
        original_settings=original_settings,
    )


def cleanup_traffic(client, world, state):
    try:
        if state is None:
            return
        print('\ndestroying %d vehicles' % len(state.vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in state.vehicles_list])

        all_actors = world.get_actors(state.all_id)
        for i in range(0, len(state.all_id), 2):
            try:
                all_actors[i].stop()
            except Exception:
                pass

        print('\ndestroying %d walkers' % len(state.walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in state.all_id])
        time.sleep(0.5)
    finally:
        restore_world_settings(world, state.original_settings)
        
# REPLACE
def replace_static_parked_cars_with_actors(client, world):
    """
    Finds static map vehicles, hides them, and spawns real carla.Actor vehicles
    in their exact locations so they can be detected by bounding box logic.
    """
    import random
    
    env_vehicle_ids = set()
    static_transforms = []
    possible_labels = ["Vehicles", "Car", "Truck", "Bus"]

    # 1. Find all baked-in map vehicles and their locations
    for label_name in possible_labels:
        if hasattr(carla.CityObjectLabel, label_name):
            label = getattr(carla.CityObjectLabel, label_name)
            for obj in world.get_environment_objects(label):
                env_vehicle_ids.add(obj.id)
                static_transforms.append(obj.transform)

    if not static_transforms:
        print("No static parked cars found on this map to replace.")
        return []

    # 2. Hide the fake static cars
    world.enable_environment_objects(env_vehicle_ids, False)

    # 3. Get safe vehicle blueprints (exclude bikes to prevent them falling over)
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    blueprints = [bp for bp in blueprints if bp.get_attribute('base_type') == 'car']

    parked_actor_ids = []
    
    # 4. Spawn real actors in those spots
    for transform in static_transforms:
        bp = random.choice(blueprints)
        bp.set_attribute('role_name', 'parked')
        
        # Give a tiny bump to the Z-axis so they don't clip into the ground and explode
        transform.location.z += 0.2  
        
        actor = world.try_spawn_actor(bp, transform)
        if actor is not None:
            # Turn off physics so it sits perfectly still and saves CPU
            actor.set_simulate_physics(False) 
            parked_actor_ids.append(actor.id)

    print(f"Replaced {len(static_transforms)} map vehicles with {len(parked_actor_ids)} detectable parked actors.")
    
    # Return the IDs so we can clean them up later
    return parked_actor_ids


def cleanup_static_parked_replacements(client, world, parked_actor_ids):
    """Destroys the spawned parked vehicles at the end of the script."""
    if not parked_actor_ids:
        return
    print(f"Cleaning up {len(parked_actor_ids)} parked vehicles...")
    client.apply_batch([carla.command.DestroyActor(x) for x in parked_actor_ids])
