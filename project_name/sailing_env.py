import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct
import jax.random as jrandom


"""
Boat angle is 0 to 360 (0 is the top)
Wind angle is -180 to 180 (0 is the top)
"""


@struct.dataclass
class BoatState:
    angle: jnp.ndarray
    speed: jnp.ndarray
    x: jnp.ndarray
    y: jnp.ndarray
    control_1: jnp.ndarray
    control_2: jnp.ndarray
    time: int


@struct.dataclass
class EnvParams:
    dt: float = 0.1
    boat_length: float = 40.0
    boat_width: float =  20.0
    turning_speed: float = 50.0  # Degrees per update
    acceleration: float = 0.1
    deceleration: float = 0.05
    max_speed: float = 5.0
    close_haul_angle: float = 30.0
    close_haul_angle_multiplier: float = 0.6
    close_haul_speed_multiplier: float = 0.4
    upwind_mark = jnp.array((20.0, 10.0))

    settings = {
        "close_hauled": {"sailing_angle": 30.0, "angle_multiplier": jnp.array(0.6), "speed_multiplier": jnp.array(0.4)},
        "beam_reach": {"angle_multiplier": jnp.array(0.3), "speed_multiplier": jnp.array(0.8)},
        "broad_reach": {"angle_multiplier": jnp.array(0.1), "speed_multiplier": jnp.array(0.6)},
    }

@struct.dataclass
class StepState:
    wind_angle: jnp.ndarray
    wind_speed: jnp.ndarray
    boat_x: jnp.ndarray
    boat_y: jnp.ndarray
    boat_speed: jnp.ndarray
    vmg_upwind: jnp.ndarray
    time: int


# Wind properties (can be functions of time for more complex scenarios)
def get_wind(time, key):
    key, _key = jrandom.split(key)
    wind_direction_deg = (20 * jnp.sin(0.1 * time)) + (jrandom.normal(_key) * 3.0)
    key, _key = jrandom.split(key)
    wind_speed = (5.0 + 1 * jnp.cos(0.05 * time) * 0.1 * jnp.sin(0.01 * time)) + (jrandom.normal(_key) * 0.1)
    return wind_direction_deg, wind_speed, key

def get_effective_speed_multiplier(wind_speed, control_1, base_multiplier):
    # TODO add some conditionals for diff wind_speed or something
    return base_multiplier * (1.2 - control_1)

def get_effective_close_haul_angle(control_1, control_2, base_angle=30.0):
    return base_angle * (1.0 - 0.2 * control_2 + 0.4 * control_1)

def update_boat_state(boat_state, key):
    wind_dir_deg, wind_strength, key = get_wind(boat_state.time, key)

    delta_angle = (wind_dir_deg - boat_state.angle + 180) % 360 - 180
    relative_wind_angle_deg = delta_angle
    # TODO check the above is correct for deep reaches and runs
    relative_wind_angle_rad = jnp.radians(relative_wind_angle_deg)

    # TODO add in apparent wind at some point
    # apparent_wind_angle_rad =

    effective_speed_multiplier = get_effective_speed_multiplier(wind_strength, boat_state.control_1, params.close_haul_speed_multiplier)
    effective_close_haul_angle = get_effective_close_haul_angle(boat_state.control_1, boat_state.control_2, params.close_haul_angle)

    potential_forward_force = wind_strength * jnp.cos(relative_wind_angle_rad) * effective_speed_multiplier
    target_speed = jnp.clip(potential_forward_force * 0.5, 0.0, params.max_speed)  # Adjust 0.5 as needed
    speed_error = target_speed - boat_state.speed
    acceleration = params.acceleration * jnp.sign(speed_error)
    delta_speed = acceleration * params.dt

    delta_speed = jnp.clip(delta_speed, -params.deceleration * params.dt * 2,
                           params.acceleration * params.dt * 2)  # Adjusted clipping

    new_boat_speed = jnp.clip(boat_state.speed + delta_speed, 0, params.max_speed)

    # Update heading (only turn if moving)
    desired_heading = wind_dir_deg + effective_close_haul_angle
    heading_difference = (desired_heading - boat_state.angle + 180) % 360 - 180
    turn = jnp.clip(heading_difference, -params.turning_speed * params.dt, params.turning_speed * params.dt)
    new_boat_heading = (boat_state.angle + turn) % 360
    boat_heading_rad = jnp.radians(new_boat_heading)
    # TODO is this okay?

    delta_x = new_boat_speed * jnp.cos(boat_heading_rad) * params.dt
    delta_y = new_boat_speed * jnp.sin(boat_heading_rad) * params.dt
    new_boat_x = boat_state.x + delta_x
    new_boat_y = boat_state.y + delta_y

    return (BoatState(angle=new_boat_heading, speed=new_boat_speed, x=new_boat_x, y=new_boat_y,
                      control_1=boat_state.control_1, control_2=boat_state.control_2, time=boat_state.time+1),
            StepState(wind_angle=wind_dir_deg, wind_speed=wind_strength, boat_x=new_boat_x, boat_y=new_boat_y,
                      boat_speed=new_boat_speed, time=boat_state.time+1, vmg_upwind=jnp.zeros(1,)),
            key)

# Simulation parameters
initial_setting = "close_hauled"
num_steps = 500
boat_start_x = jnp.array((0.0,))
boat_start_y = jnp.array((0.0,))
boat_start_control_1 = jnp.array((0.0,))
boat_start_control_2 = jnp.array((0.0,))
initial_boat_heading = jnp.array((30.0,))
initial_boat_speed = jnp.array((1.0,))
params = EnvParams()
key = jrandom.key(42)
initial_state = BoatState(angle=initial_boat_heading, speed=initial_boat_speed,
                          x=boat_start_x, y=boat_start_y, control_1=boat_start_control_1, control_2=boat_start_control_2,
                          time=0)

def _step(runner_state, unused):
    current_state, key = runner_state
    key, _key = jrandom.split(key)
    next_boat_state, step_state, key = update_boat_state(current_state, _key)

    north_direction_rad = jnp.radians(0.0)
    north_unit_vector = jnp.array([jnp.cos(north_direction_rad), jnp.sin(north_direction_rad)])
    boat_heading_rad = jnp.radians(next_boat_state.angle)
    boat_velocity = next_boat_state.speed * jnp.array([jnp.cos(boat_heading_rad), jnp.sin(boat_heading_rad)])

    # Calculate VMG upwind
    vmg_upwind = jnp.dot(jnp.squeeze(boat_velocity, axis=-1), north_unit_vector)

    step_state = step_state.replace(vmg_upwind=vmg_upwind)

    return (next_boat_state, key), step_state

with jax.disable_jit(disable=False):
    last_state, simulation_history = jax.lax.scan(_step, (initial_state, key), None, num_steps)

fig, axs = plt.subplots(2, 1, figsize=(10, 12))
axs[0].plot(simulation_history.time, simulation_history.wind_speed)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Wind Speed")
axs[0].set_title("Wind Speed Over Time")
axs[0].grid(True)

axs[1].plot(simulation_history.wind_angle, simulation_history.time)
axs[1].set_ylabel("Time")
axs[1].set_xlabel("Wind Direction (Degrees)")
axs[1].set_title("Wind Direction Over Time")
axs[1].grid(True)
axs[1].set_xticks(jnp.arange(-180, 181, 45))

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].plot(simulation_history.boat_x, simulation_history.boat_y)
axs[0].set_xlabel("X Position")
axs[0].set_ylabel("Y Position")
axs[0].set_title("Boat Trajectory")
axs[0].set_xlim(0, 50)
axs[0].set_ylim(0, 30)
axs[0].grid(True)
# axs[0].axis('equal') # Ensure proper aspect ratio for trajectory

axs[1].plot(simulation_history.time, simulation_history.boat_speed)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Boat Speed")
axs[1].set_title("Boat Speed Over Time")
axs[1].grid(True)

axs[2].plot(simulation_history.time, simulation_history.vmg_upwind)
axs[2].set_xlabel("Time")
axs[2].set_ylabel("VMG Upwind")
axs[2].set_title("Velocity Made Good (VMG) to 0 degrees")
axs[2].grid(True)

plt.tight_layout()
plt.show()