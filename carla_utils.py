import os
import random
from typing import Dict

import carla

from carla_config import SimulationConfig


def ensure_directories(cfg):
    for path in [cfg.capture.output_dir, cfg.capture.labels_dir, cfg.capture.rgb_dir, cfg.capture.bbox_dir]:
        os.makedirs(path, exist_ok=True)


def build_projection_matrix(width, height, fov):
    import numpy as np

    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def set_weather(world, weather_cfg):
    weather = carla.WeatherParameters()
    weather.cloudiness = weather_cfg.cloudiness
    weather.precipitation = weather_cfg.precipitation
    weather.precipitation_deposits = weather_cfg.precipitation_deposits
    weather.wind_intensity = weather_cfg.wind_intensity
    weather.sun_azimuth_angle = weather_cfg.sun_azimuth_angle
    weather.sun_altitude_angle = weather_cfg.sun_altitude_angle
    weather.fog_density = weather_cfg.fog_density
    weather.fog_distance = weather_cfg.fog_distance
    weather.fog_falloff = weather_cfg.fog_falloff
    weather.wetness = weather_cfg.wetness
    weather.scattering_intensity = weather_cfg.scattering_intensity
    weather.mie_scattering_scale = weather_cfg.mie_scattering_scale
    weather.rayleigh_scattering_scale = weather_cfg.rayleigh_scattering_scale
    weather.dust_storm = weather_cfg.dust_storm
    world.set_weather(weather)

def load_world(client, map_name):
    """Load a CARLA map if configured, otherwise keep the current world."""
    if map_name:
        return client.load_world(map_name)
    return client.get_world()

def apply_world_settings(world, cfg, traffic_manager=None):
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = cfg.synchronous_mode
    settings.fixed_delta_seconds = cfg.fixed_delta_seconds if cfg.synchronous_mode else None
    settings.no_rendering_mode = cfg.no_rendering_mode
    world.apply_settings(settings)

    synchronous_master = False
    if traffic_manager is not None and cfg.synchronous_mode:
        traffic_manager.set_synchronous_mode(True)
        synchronous_master = not original_settings.synchronous_mode
    elif traffic_manager is not None:
        traffic_manager.set_synchronous_mode(False)

    return original_settings, synchronous_master


def restore_world_settings(world, original_settings):
    world.apply_settings(original_settings)


def resolve_seeds(cfg):
    if cfg.master_seed is None and cfg.traffic_seed is None and cfg.ego_seed is None and cfg.walker_seed is None:
        master = random.SystemRandom().randint(0, 2**31 - 1)
    else:
        master = cfg.master_seed if cfg.master_seed is not None else random.SystemRandom().randint(0, 2**31 - 1)

    traffic_seed = cfg.traffic_seed if cfg.traffic_seed is not None else master
    ego_seed = cfg.ego_seed if cfg.ego_seed is not None else master + 1
    walker_seed = cfg.walker_seed if cfg.walker_seed is not None else master + 2

    return {
        "master_seed": master,
        "traffic_seed": traffic_seed,
        "ego_seed": ego_seed,
        "walker_seed": walker_seed,
    }


def choose_spawn_point(spawn_points, seed):
    rng = random.Random(seed)
    return rng.choice(spawn_points)
