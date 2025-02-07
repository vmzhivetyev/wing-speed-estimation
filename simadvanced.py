# aircraft.py
import math

import settings
# from settings import *
import numpy as np



class SimAdvanced_:
    def __init__(self, in_pitch_offset=0, in_thrust=4, in_prop_pitch=4.7, in_drag_k=0.0045):
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.mass = settings.mass
        self.twr = in_thrust / settings.mass
        self.prop_pitch_max_speed = 0.0004233 * in_prop_pitch * settings.motor_kv * settings.max_voltage
        self.drag_coefficient = in_drag_k
        self.max_voltage = settings.max_voltage
        # self.max_speed = TODO
        self.thrust = self.twr * self.mass * self.g
        self.pitch_offset = in_pitch_offset

    def get_acceleration(self, speed, roll, pitch, rpms, voltage, total_current, throttle):
        scaled_throttle = throttle * voltage / self.max_voltage
        corrected_pitch = pitch + math.cos(pitch) * math.cos(roll) * self.pitch_offset / 180 * math.pi
        drag_force = self.drag_coefficient * speed**2
        thrust_coef = scaled_throttle ** 2 - scaled_throttle * speed / self.prop_pitch_max_speed
        thrust_force = thrust_coef * self.thrust
        gravity_force = - self.mass * self.g * math.sin(corrected_pitch)
        net_force = thrust_force - drag_force - gravity_force
        a = net_force / self.mass
        return a


class SimAdvanced:
    def __init__(self, in_pitch_offset=0, in_thrust=4, in_prop_pitch=2.4, in_drag_k=0.0045):
        self.g = 9.81  # Gravity (m/s^2)
        self.mass = settings.mass
        self.twr = in_thrust / self.mass
        self.prop_pitch = in_prop_pitch  # Inches
        self.drag_coefficient = in_drag_k
        self.motor_kv = settings.motor_kv  # RPM per volt
        self.max_voltage = settings.max_voltage  # 4S LiPo
        self.max_rpm = self.motor_kv * self.max_voltage  # Theoretical max RPM
        self.pitch_offset = in_pitch_offset

    def step(self, speed, roll, pitch, rpms, voltage, total_current, throttle):
        standby_current = 0.5
        all_motors_current = max(0, total_current - standby_current)

        # Convert RPM to airflow speed
        assert len(rpms) == 2
        avg_rpm = np.mean(rpms)

        # Handle near-zero RPM cases
        if avg_rpm < 500:
            thrust_force = 0
        else:
            # Compute thrust using quadratic approximation
            prop_speed_mps = avg_rpm * self.prop_pitch * 0.0254 / 60  # Convert pitch in inches to meters per sec
            thrust_coef = (prop_speed_mps - speed) / prop_speed_mps
            thrust_force = max(0, thrust_coef * self.twr * self.mass * self.g)

        # Compute drag
        drag_force = - self.drag_coefficient * speed ** 2

        # Compute gravity effect
        corrected_pitch = math.degrees(pitch) + self.pitch_offset
        gravity_force = self.mass * self.g * math.sin(math.radians(corrected_pitch)) * math.cos(math.radians(roll))

        # Optional: Consider electrical power losses if current is provided
        # Rough estimate of power loss: resistive + inefficiencies
        V_motor = throttle * voltage  # Effective voltage applied to motors

        input_power = V_motor * all_motors_current  # Power actually going to the motors
        power_loss = 0.01 * all_motors_current ** 2  # Example resistive + inefficiency losses
        loss_thrust_coeff = np.clip(1 - power_loss / max(input_power, 1e-2), 0, 1)
        thrust_force *= loss_thrust_coeff  # Apply scaling

        # Compute acceleration
        net_force = thrust_force + drag_force + gravity_force
        a = net_force / self.mass
        assert speed < 200 / 3.6
        return a, thrust_force, drag_force, gravity_force, input_power, power_loss


'''

wind 2m/s

ADVANCED optimization time: 0 hours, 7 minutes, 59.49 seconds
ADVANCED Optimal Parameters: [-1.22542714e+01  1.20000000e+00  2.40000000e+00  1.00000000e-03]
ADVANCED Minimum Value: 8.024283916189086

XXXX print_cli_settings: PID: 46002, Called print_cli_settings()
#========================================
set tpa_speed_type = ADVANCED
set tpa_speed_adv_mass = 270
set tpa_speed_adv_drag_k = 10
set tpa_speed_adv_thrust = 1200
set tpa_speed_adv_prop_pitch = 240
set motor_kv = 4500
set tpa_speed_pitch_offset = -123
#========================================




ADVANCED optimization time: 0 hours, 1 minutes, 2.83 seconds
ADVANCED Optimal Parameters: [-12.4500000  1.53000000  2.40000000  1.70473008e-03]
ADVANCED Minimum Value: 7.665199956918219

XXXX print_cli_settings: PID: 48682, Called print_cli_settings()
#========================================
set tpa_speed_type = ADVANCED
set tpa_speed_adv_mass = 270
set tpa_speed_adv_drag_k = 17
set tpa_speed_adv_thrust = 1530
set tpa_speed_adv_prop_pitch = 240
set motor_kv = 4500
set tpa_speed_pitch_offset = -124
#========================================
'''