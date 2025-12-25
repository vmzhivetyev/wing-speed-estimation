import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import numpy as np
import os

import math
from functools import partial
import time

from csv_reader import read_csv_as_dict
# from sim_basic import *
from simadvanced import *
from headers import *
from utils import *
import settings
import sys


def evaluate(sim, bbx_loop_range, needs_results, context):
    error = 0
    count = 0

    dt, pitches, rolls, throttles, voltages, gps_speeds, currents, rpms, elevators = context
    v = next(x for x in gps_speeds if not math.isnan(x))

    v_results = []
    steps = []

    for i in bbx_loop_range:
        step_data = sim.step(v, rolls[i], pitches[i], rpms[i], voltages[i], currents[i], throttles[i], elevators[i])
        steps.append(step_data)
        a, thrust_force, drag_force, elevator_drag_force, gravity_force, input_power, power_loss = step_data
        v = v + a * dt
        v = max(v, 0)
        if needs_results:
            v_results.append(v)

        d_error = (v - gps_speeds[i]) ** 2
        if not math.isnan(d_error):
            error = error + d_error
            count = count + 1

    return error / count, v_results, steps


def get_error_sim_advanced(params, bbx_loop_range, context):
    # print(f'get_error_sim_advanced {params}')
    param_pitch_offset, param_thrust, param_prop_pitch, param_drag_coefficient = params
    config = AdvancedConfig()
    config.pitch_offset_advanced = param_pitch_offset
    config.thrust = param_thrust
    config.prop_pitch = param_prop_pitch
    config.drag_k = param_drag_coefficient

    sim = SimAdvanced(config=config)
    t = time.time()
    error, _, _ = evaluate(sim, bbx_loop_range, needs_results=False, context=context)
    e = time.time()
    # Assuming `params` is a list or array of float values
    formatted_params = ' '.join(f'{p:8.5f}' for p in params)  # 8.2f gives 2 decimal points with a width of 8 characters

    # Print the formatted error and parameters
    print(f'get_error_sim_advanced error: {error:6.2f} params: {formatted_params} took: {e - t:0.3f}s')
    return error


# def get_error_sim_basic(params, data_dict, bbx_loop_range):
#     param_pitch_offset, param_gravity, param_delay = params
#     sim = Sim_basic(param_pitch_offset, param_gravity, param_delay)
#     return get_error(sim, data_dict, bbx_loop_range)


def progress_callback(xk, convergence):
    # Log progress here. You can print or store information about the optimization.
    print(f"Current best solution: {xk}, Convergence: {convergence}")


def get_optimal_params(get_error_with_data, bounds, sim_name):
    start_time = time.time()
    result = differential_evolution(
        get_error_with_data,
        bounds,
        workers=1,
        callback=progress_callback,
        strategy='randtobest1bin'
    )
    seconds = time.time() - start_time
    print(f"{sim_name} optimization time: {format_seconds(seconds)}")
    minimum_value = result.fun
    print(f"{sim_name} Optimal Parameters:", result.x)
    print(f"{sim_name} Minimum Value:", minimum_value)
    return result.x


print(f"XXXX Before main print_cli_settings: PID: {os.getpid()}, Called print_cli_settings()")

# def get_error_with_data(params):
#     print(f'get_error_with_data 🫠 {params}')
#     return get_error_sim_advanced(params, data_dict=data_dict, bbx_loop_range=bbx_loop_range)


def make_context(data_dict, bbx_loop_range):
    dt = data_dict['dt'] * bbx_loop_range.step
    pitches = data_dict[header_pitch]
    rolls = data_dict[header_roll]
    throttles = data_dict['Throttle']
    voltages = data_dict[header_voltage]
    gps_speeds = data_dict[header_gps_speed]
    currents = data_dict['amperageLatest']
    rpms_keys = [x for x in data_dict.keys() if x.startswith("true_rpm")]
    rpms = [data_dict[key] for key in rpms_keys]
    rpms = np.transpose(rpms)
    elevators = (data_dict['servo[2]'] - 1500) / 500
    return dt, pitches, rolls, throttles, voltages, gps_speeds, currents, rpms, elevators


def plot_data(data_dict, sim_data, ax_gps_speed, ax_forces, ax_throttle, colorset, suffix: str):
    error_data, data_sim_advanced, steps_sim_advanced = sim_data

    data_time = data_dict[header_time]  # Time in seconds
    data_gps_speed = data_dict[header_gps_speed]  # GPS speed values
    data_throttle = data_dict['Throttle']  # Throttle values
    data_pitch_degrees = data_dict[header_pitch] * 180 / math.pi  # Pitch values

    valid_gps_speeds = [val for val in data_gps_speed if not math.isnan(val)]

    # plt.subplots_adjust(hspace=0)

    colors_gps_speed = colorset[0]  # 'red'
    colors_sim_speed = colorset[1]  # 'tab:blue'
    throttle_color = colorset[2]  # 'orange'
    lw = 1

    data_gps_speed = np.multiply(data_gps_speed, 3.6)
    data_sim_advanced = np.multiply(data_sim_advanced, 3.6)
    max_y_value = max(max(valid_gps_speeds), max(data_sim_advanced)) * 1.2
    max_y_value = min(max_y_value, 200)

    # First plot (GPS Speed over Time)
    label_gps_speed = f'GPS Speed (m/s)'
    line_gps_speed = ax_gps_speed.plot(data_time, data_gps_speed, label=label_gps_speed, color=colors_gps_speed, lw=lw)[0]
    ax_gps_speed.set_ylabel(label_gps_speed, color=colors_gps_speed)
    ax_gps_speed.tick_params(axis='y', labelcolor=colors_gps_speed)
    ax_gps_speed.grid(True)

    ax_sim_speed = ax_gps_speed.twinx()
    ax_sim_speed.spines['right'].set_position(('outward', 20))
    line_simulation2 = ax_sim_speed.plot(
        data_time,
        data_sim_advanced,
        label=f'Advanced Simulation {suffix}',
        color=colors_sim_speed
    )[0]
    ax_sim_speed.set_ylabel(f'Advanced Simulation {suffix}', color=colors_sim_speed)
    ax_sim_speed.tick_params(axis='y', labelcolor=colors_sim_speed)

    lines = [line_gps_speed, line_simulation2]
    labels = [line.get_label() for line in lines]
    ax_gps_speed.legend(lines, labels, loc='upper right')

    ax_gps_speed.set_ylim(0, max_y_value)
    ax_sim_speed.set_ylim(0, max_y_value)

    # AX BTW
    a, thrust_force, drag_force, elevator_drag_force, gravity_force, input_power, power_loss = np.transpose(steps_sim_advanced)
    # ax_forces.plot(data_time, thrust_force, label='thrust_force')
    ax_forces.plot(data_time, drag_force, label=f'drag_force {suffix}')
    ax_forces.plot(data_time, elevator_drag_force, label=f'elevator_drag_force {suffix}')
    # ax_forces.plot(data_time, gravity_force, label='gravity_force')
    ax_forces.plot(data_time, np.sum((thrust_force, drag_force, gravity_force), axis=0), label='total')
    ax_forces.legend()
    ax_forces.grid(True)

    # Second plot (Throttle and Pitch over Time)
    ax_throttle.plot(data_time, data_throttle, throttle_color, label='Throttle')  # Plot throttle in red
    ax_throttle.set_ylabel('Throttle', color=throttle_color)
    ax_throttle.tick_params(axis='y', colors=throttle_color)
    ax_throttle.grid(axis='y', linestyle=':', color=throttle_color)  # Make only throttle horizontal grid marks red and dotted
    ax_throttle.grid(axis='x')  # Keep vertical lines the same as the top plot
    ax_throttle.set_ylim(0, 1.1)

    # Create a second Y-axis on the right for pitch
    pitch_color = 'green'
    ax4 = ax_throttle.twinx()
    ax4.plot(data_time, data_pitch_degrees, pitch_color, label=f'Pitch {suffix}')  # Plot pitch in blue
    ax4.set_ylabel('Pitch', color=pitch_color)
    ax4.tick_params(axis='y', colors=pitch_color)
    ax4.set_ylim(-100, 100)
    ax4.axhline(0, linestyle=':', color=pitch_color)  # Add a dotted blue line at pitch = 0

    # Align the legends
    # ax2.legend(loc='upper left')
    # ax3.legend(loc='upper right')


def main():
    # Example usage of the function
    print(f"XXXX __main__ print_cli_settings: PID: {os.getpid()}, Called print_cli_settings()")

    # get_error_sim_advanced error:  19.83 params: -10.00000  1.00000  2.40000  0.00235 took: 0.147s
    file_path = 'logs/speed_foam/dive.csv'

    file_path = 'logs/speed_foam/btfl_006.bbl.csv'

    # file_path = select_csv_file("logs")
    data_dict = read_csv_as_dict(file_path)

    bbx_loop_range = calculate_bbx_loop_range(data_dict=data_dict)
    context = make_context(data_dict, bbx_loop_range)

    config = settings.AdvancedConfig()

    if settings.calculate:
        calc_config = CalculationConfig()
        print("Running differential_evolution for ADVANCED")
        bounds = [calc_config.range_pitch_offset, calc_config.range_thrust, calc_config.range_prop_pitch, calc_config.range_drag_k]
        get_error_with_data = partial(get_error_sim_advanced, bbx_loop_range=bbx_loop_range, context=context)
        optimal_params = get_optimal_params(get_error_with_data, bounds, "ADVANCED")
        new_pitch_offset_advanced, new_thrust, new_prop_pitch, new_drag_k = optimal_params
        config.pitch_offset_advanced = new_pitch_offset_advanced
        config.thrust = new_thrust
        config.prop_pitch = new_prop_pitch
        config.drag_k = new_drag_k

    conf1 = config
    sim1 = evaluate(SimAdvanced(config=conf1), bbx_loop_range, needs_results=True, context=context)

    conf2 = config
    conf2.drag_k_elevator = 0.005
    sim2 = evaluate(SimAdvanced(config=conf2), bbx_loop_range, needs_results=True, context=context)

    CLISettingsPrinter.print_cli_settings(advanced=conf1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [14, 3, 3]})
    plot_data(data_dict, sim1, axes[0], axes[1], axes[2], colorset=['C0', 'C1', 'C2'], suffix='base')  # *np.transpose(axes)[0]
    plot_data(data_dict, sim2, axes[0], axes[1], axes[2], colorset=['C0', 'C3', 'C4'], suffix='less drag')  # *np.transpose(axes)[0]

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, left=0.04, right=1, wspace=0.1)

    plt.show()


if __name__ == '__main__':
    main()
