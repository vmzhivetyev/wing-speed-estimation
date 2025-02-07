import math
import os

# change this between True/False if you need to run automatic parameters finding
calculate = False

# user defined parameters
max_voltage = 4.2 * 4  # volts
mass = 0.270  # kilograms
motor_kv = 4500

# ranges of parameters for the automatic algorithm to find best values

# old
range_pitch_offset = [-15, -5]  # degrees
# range_pitch_offset = [-12.45, -12.45]  # degrees

# range_delay = (0.1, 2)  # seconds
# range_gravity = (0.1, 10)  # percents / 100

# range_thrust = (1.53, 1.53)  # kilograms
range_thrust = (1.2, 1.7)  # kilograms

range_prop_pitch = (2.4, 2.4)  # inches
range_drag_k = (0.0001, 0.01)


# parameters to plot with, when automatic parameters finding is False (OFF)
prop_pitch = 2.4  # inches
drag_k = 0.0015
thrust = 0.8  # kilograms
pitch_offset_basic = 0  # degrees
pitch_offset_advanced = -15  # degrees

tpa_delay = 0.5
tpa_gravity = 0.5

dt_target = 0.01
time_start = 0
time_stop = 999


def print_cli_settings():
    settings_text = (
        f"\n"
        f"XXXX print_cli_settings: PID: {os.getpid()}, Called print_cli_settings()\n"
        # f"#========================================\n"
        # f"set tpa_speed_type = BASIC\n"
        # f"set tpa_speed_basic_delay = {round(tpa_delay * 1000)}\n"
        # f"set tpa_speed_basic_gravity = {round(tpa_gravity * 100)}\n"
        # # f"set tpa_speed_max_voltage = {round(max_voltage * 100)}\n"
        # f"set tpa_speed_pitch_offset = {round(pitch_offset_basic * 10)}\n"
        f"#========================================\n"
        f"set tpa_speed_type = ADVANCED\n"
        f"set tpa_speed_adv_mass = {round(mass * 1000)}\n"
        f"set tpa_speed_adv_drag_k = {round(drag_k * 10000)}\n"
        f"set tpa_speed_adv_thrust = {round(thrust * 1000)}\n"
        f"set tpa_speed_adv_prop_pitch = {round(prop_pitch * 100)}\n"
        f"set motor_kv = {round(motor_kv)}\n"
        # f"set tpa_speed_max_voltage = {round(max_voltage * 100)}\n"
        f"set tpa_speed_pitch_offset = {round(pitch_offset_advanced * 10)}\n"
        f"#========================================\n"
    )
    print(settings_text)
