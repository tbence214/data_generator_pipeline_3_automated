from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeatherConfig:
    cloudiness: float = 0.0 # 0 = clear sky, 100 = fully overcast
    precipitation: float = 0.0 # Rain intensity. 0 = none, 100 = heavy rain.
    precipitation_deposits: float = 0.0 # Puddles on the road. 0 = none, 100 = road covered with water.
    wind_intensity: float = 20.0 # Wind strength. It mainly matters for the look of rain and tree motion.
    sun_azimuth_angle: float = 0.0 # Turns the sun around the horizon. It changes the direction of sunlight and shadows.
    sun_altitude_angle: float = 70.0 # 90 is midday, 0 is the horizon, and negative values are below the horizon.
    fog_density: float = 0.0
    fog_distance: float = 0.0
    fog_falloff: float = 0.0
    wetness: float = 0.0 # Road wetness.
    scattering_intensity: float = 1.0 # How much light contributes to volumetric fog. Higher values make fog feel more present.
    mie_scattering_scale: float = 0.03 # Creates haze and halos around light sources. Higher values make the sky and air look more polluted or hazy.
    rayleigh_scattering_scale: float = 0.0331 # Controls the blue-sky / red-sunset atmosphere. Higher or lower values can shift the feel of daylight quite a bit. CARLA’s default constructor value is 0.0331.
    dust_storm: float = 0.0 # Dust storm strength, from 0 to 100.


@dataclass
class TrafficConfig:
    number_of_vehicles: int = 40
    number_of_walkers: int = 40
    safe: bool = False
    filterv: str = "vehicle.*"
    generationv: str = "All"
    filterw: str = "walker.pedestrian.*"
    generationw: str = "2"
    tm_port: int = 8000
    asynch: bool = False
    hybrid: bool = False
    car_lights_on: bool = False
    hero: bool = False
    respawn: bool = False
    no_rendering: bool = False


@dataclass
class CaptureConfig:
    max_frames: int = 1000
    img_width: int = 1920
    img_height: int = 1080
    fov: float = 90.0
    min_box_area: int = 150
    depth_tolerance_meters: float = 2.5
    min_visible_ratio: float = 0.35
    visible_sample_grid: int = 5
    draw_labels: bool = False
    output_dir: str = "output"
    labels_dir: str = "output/labels"
    rgb_dir: str = "output/pictures"
    bbox_dir: str = "output/b_picture"


@dataclass
class SimulationConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    map_name: Optional[str] = None # "Town03" Set to None to keep the current CARLA map.
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.05
    no_rendering_mode: bool = False
    ego_blueprint: str = "vehicle.tesla.model3"
    ego_role_name: str = "ego"
    ego_autopilot: bool = True
    ego_camera_x: float = 1.5
    ego_camera_y: float = 0.0
    ego_camera_z: float = 1.2
    master_seed: Optional[int] = 40
    traffic_seed: Optional[int] = 40
    ego_seed: Optional[int] = 40
    walker_seed: Optional[int] = 40
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
