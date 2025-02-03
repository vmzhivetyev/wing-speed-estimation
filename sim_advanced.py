# aircraft.py
import math
from settings import *
import numpy as np

class Sim_advanced:
    def __init__(self, in_pitch_offset=0, in_thrust=4, in_prop_pitch=4.7, in_drag_k=0.0045):
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.mass = mass
        self.twr = in_thrust / mass
        self.prop_pitch_max_speed = 0.0004233 * in_prop_pitch * motor_kv * max_voltage
        self.drag_coefficient = in_drag_k
        self.max_voltage = max_voltage
        # self.max_speed = TODO
        self.thrust = self.twr * self.mass * self.g
        self.pitch_offset = in_pitch_offset

    def get_acceleration(self, speed, roll, pitch, throttle, voltage):
        scaled_throttle = throttle * voltage / self.max_voltage
        corrected_pitch = pitch + math.cos(pitch) * math.cos(roll) * self.pitch_offset / 180 * math.pi
        drag_force = self.drag_coefficient * speed**2
        thrust_coef = scaled_throttle ** 2 - scaled_throttle * speed / self.prop_pitch_max_speed
        thrust_force = thrust_coef * self.thrust
        gravity_force = - self.mass * self.g * math.sin(corrected_pitch)
        net_force = thrust_force - drag_force - gravity_force
        a = net_force / self.mass
        return a


'''

wind 2m/s

Running differential_evolution for ADVANCED
ADVANCED optimization time: 0 hours, 0 minutes, 51.54 seconds
ADVANCED Optimal Parameters: [1.16265692e+01 1.20000000e+00 2.40000000e+00 2.23154066e-03]
ADVANCED Minimum Value: 27.55438155797812
Running differential_evolution for BASIC
BASIC optimization time: 0 hours, 0 minutes, 54.02 seconds
BASIC Optimal Parameters: [8.45034719 0.32015305 0.2892114 ]
BASIC Minimum Value: 12.667762612024317

XXXX print_cli_settings: PID: 34040, Called print_cli_settings()
#========================================
set tpa_speed_type = BASIC
set tpa_speed_basic_delay = 289
set tpa_speed_basic_gravity = 32
set tpa_speed_pitch_offset = 85
#========================================
set tpa_speed_type = ADVANCED
set tpa_speed_adv_mass = 270
set tpa_speed_adv_drag_k = 22
set tpa_speed_adv_thrust = 1200
set tpa_speed_adv_prop_pitch = 240
set motor_kv = 4500
set tpa_speed_pitch_offset = 116
#========================================
'''