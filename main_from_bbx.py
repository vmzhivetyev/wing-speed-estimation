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
    v = 0
    error = 0
    count = 0

    dt, pitches, rolls, throttles, voltages, gps_speeds, currents, rpms = context

    v_results = []

    for i in bbx_loop_range:
        v = v + sim.get_acceleration(v, rolls[i], pitches[i], rpms[i], voltages[i], currents[i], throttles[i]) * dt
        v = max(v, 0)
        if needs_results:
            v_results.append(v)

        d_error = (v - gps_speeds[i]) ** 2
        if not math.isnan(d_error):
            error = error + d_error
            count = count + 1

    return error / count, v_results


def get_error_sim_advanced(params, bbx_loop_range, context):
    # print(f'get_error_sim_advanced {params}')
    param_pitch_offset, param_thrust, param_prop_pitch, param_drag_coefficient = params
    sim = SimAdvanced(param_pitch_offset, param_thrust, param_prop_pitch, param_drag_coefficient)
    t = time.time()
    error, _ = evaluate(sim, bbx_loop_range, needs_results=False, context=context)
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
    return dt, pitches, rolls, throttles, voltages, gps_speeds, currents, rpms


if __name__ == '__main__':
    # Example usage of the function
    print(f"XXXX __main__ print_cli_settings: PID: {os.getpid()}, Called print_cli_settings()")

    file_path = 'logs/speed_foam/btfl_006.bbl.csv'
    # file_path = select_csv_file("logs")
    data_dict = read_csv_as_dict(file_path)

    bbx_loop_range = calculate_bbx_loop_range(data_dict=data_dict)
    context = make_context(data_dict, bbx_loop_range)

    if settings.calculate:
        print("Running differential_evolution for ADVANCED")
        bounds = [range_pitch_offset, range_thrust, range_prop_pitch, range_drag_k]
        get_error_with_data = partial(get_error_sim_advanced, bbx_loop_range=bbx_loop_range, context=context)
        optimal_params = get_optimal_params(get_error_with_data, bounds, "ADVANCED")
        settings.pitch_offset_advanced, settings.thrust, settings.prop_pitch, settings.drag_k = optimal_params

    # if settings.calculate:
    #     print("Running differential_evolution for BASIC")
    #     bounds = [range_pitch_offset, range_gravity, range_delay]
    #     get_error_with_data = partial(get_error_sim_basic, data_dict=data_dict, bbx_loop_range=bbx_loop_range)
    #     optimal_params = get_optimal_params(get_error_with_data, bounds, "BASIC")
    #     settings.pitch_offset_basic, settings.tpa_gravity, settings.tpa_delay = optimal_params

    sim_advanced = SimAdvanced(in_pitch_offset=settings.pitch_offset_advanced, in_thrust=settings.thrust,
                               in_prop_pitch=settings.prop_pitch, in_drag_k=settings.drag_k)
    # sim_basic = Sim_basic(in_pitch_offset=settings.pitch_offset_basic, in_gravity=settings.tpa_gravity, in_delay=settings.tpa_delay)

    # error_basic = get_error(sim_basic, data_dict, bbx_loop_range)
    error_advanced, data_sim_advanced = evaluate(sim_advanced, bbx_loop_range, needs_results=True, context=context)

    data_time = data_dict[header_time]  # Time in seconds
    data_gps_speed = data_dict[header_gps_speed]  # GPS speed values
    data_throttle = data_dict['Throttle']  # Throttle values
    data_pitch_degrees = data_dict[header_pitch] * 180 / math.pi  # Pitch values

    valid_gps_speeds = [val for val in data_gps_speed if not math.isnan(val)]
    max_y_value = np.max(valid_gps_speeds + data_sim_advanced)  # data_sim_basic

    # Create the figure
    fig, (ax_gps_speed, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [7, 3]})
    fig.subplots_adjust(right=0.8)

    # Adjust the space between plots to zero
    plt.subplots_adjust(hspace=0)

    # First plot (GPS Speed over Time)
    line_gps_speed = ax_gps_speed.plot(data_time, data_gps_speed, label='GPS Speed', color='r')[0]
    ax_gps_speed.set_ylabel('GPS Speed', color='r')
    ax_gps_speed.set_title('GPS Speed over Time')
    ax_gps_speed.tick_params(axis='y', labelcolor='r')
    ax_gps_speed.grid(True)

    # Create a secondary y-axis for the simple simulation
    # ax_simulation1 = ax_gps_speed.twinx()
    # line_simulation1 = ax_simulation1.plot(data_time, data_sim_basic, label=f'Basic Simulation\n(err = {math.sqrt(error_basic / math.pi * 2):.2f})', color='g')[0]
    # ax_simulation1.set_ylabel('Simple Simulation', color='g')
    # ax_simulation1.tick_params(axis='y', labelcolor='g')

    ax_simulation2 = ax_gps_speed.twinx()
    ax_simulation2.spines['right'].set_position(('outward', 60))
    line_simulation2 = ax_simulation2.plot(data_time, data_sim_advanced,
                                           label=f'Advanced Simulation\n(err = {math.sqrt(error_advanced / math.pi * 2):.2f})',
                                           color='b')[0]
    ax_simulation2.set_ylabel('Advanced Simulation', color='b')
    ax_simulation2.tick_params(axis='y', labelcolor='b')

    lines = [line_gps_speed, line_simulation2]
    labels = [line.get_label() for line in lines]
    ax_gps_speed.legend(lines, labels, loc='upper left')

    ax_gps_speed.set_ylim(0, max_y_value)
    # ax_simulation1.set_ylim(0, max_y_value)
    ax_simulation2.set_ylim(0, max_y_value)

    # Second plot (Throttle and Pitch over Time)
    ax3.plot(data_time, data_throttle, 'r', label='Throttle')  # Plot throttle in red
    ax3.set_ylabel('Throttle', color='red')
    ax3.tick_params(axis='y', colors='red')
    ax3.grid(axis='y', linestyle=':', color='red')  # Make only throttle horizontal grid marks red and dotted
    ax3.grid(axis='x')  # Keep vertical lines the same as the top plot
    ax3.set_ylim(0, 1.1)

    # Create a second Y-axis on the right for pitch
    ax4 = ax3.twinx()
    ax4.plot(data_time, data_pitch_degrees, 'b', label='Pitch')  # Plot pitch in blue
    ax4.set_ylabel('Pitch', color='blue')
    ax4.tick_params(axis='y', colors='blue')
    ax4.set_ylim(-100, 100)
    ax4.axhline(0, linestyle=':', color='blue')  # Add a dotted blue line at pitch = 0

    # Align the legends
    # ax2.legend(loc='upper left')
    # ax3.legend(loc='upper right')

    # Show the plot
print_cli_settings()
plt.tight_layout()
plt.show()
