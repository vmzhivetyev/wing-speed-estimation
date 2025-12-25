import math
import os

# change this between True/False if you need to run automatic parameters finding
calculate = False


class CalculationConfig:
    def __init__(self):
        self.range_pitch_offset = [-20, -20]  # degrees
        self.range_thrust = (0.8, 1.2)  # kilograms
        self.range_prop_pitch = (2.4, 2.4)  # inches
        self.range_drag_k = (0.00001, 0.01)


class StaticEvaluationConstants:
    def __init__(self):
        # self.tpa_delay = 0.5
        # self.tpa_gravity = 0.5
        self.dt_target = 0.01
        self.time_start = 0
        self.time_stop = 999


class BasicConfig:
    def __init__(self):
        self.mass = 0.270  # kilograms
        self.motor_kv = 4500
        self.max_voltage = 4.2 * 4  # volts


class AdvancedConfig:
    def __init__(self, basic_config=BasicConfig()):
        self.mass = basic_config.mass
        self.motor_kv = basic_config.motor_kv
        self.max_voltage = basic_config.max_voltage
        self.prop_pitch = 2.4  # inches
        self.drag_k = 0.001
        self.drag_k_elevator = 0.000
        self.thrust = 0.9  # kilograms
        self.pitch_offset_advanced = -15


class CLISettingsPrinter:
    @staticmethod
    def print_cli_settings(advanced: AdvancedConfig):
        settings_text = (
            f"\n"
            f"XXXX print_cli_settings: PID: {os.getpid()}, Called print_cli_settings()\n"
            f"#========================================\n"
            f"set tpa_speed_type = ADVANCED\n"
            f"set tpa_speed_adv_mass = {round(advanced.mass * 1000)}\n"
            f"set tpa_speed_adv_drag_k = {round(advanced.drag_k * 10000)}\n"
            f"set tpa_speed_adv_thrust = {round(advanced.thrust * 1000)}\n"
            f"set tpa_speed_adv_prop_pitch = {round(advanced.prop_pitch * 100)}\n"
            f"set motor_kv = {round(advanced.motor_kv)}\n"
            f"set tpa_speed_pitch_offset = {round(advanced.pitch_offset_advanced * 10)}\n"
            f"#========================================\n"
        )
        print(settings_text)
