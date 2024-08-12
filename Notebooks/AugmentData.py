import numpy as np
import pynumdiff
from CorrectHeading import *

def augment_df(fly_trajectory_and_body):
    gnd_velocity_x, gnd_velocity_y = fly_trajectory_and_body["velocity_x"],fly_trajectory_and_body["velocity_y"]
    fly_trajectory_and_body["groundspeed"] = np.sqrt(gnd_velocity_x**2 + gnd_velocity_y**2)
    fly_trajectory_and_body["groundspeed_angle"] = np.arctan2(gnd_velocity_y,gnd_velocity_x)
    
    airspeed_x, airspeed_y = fly_trajectory_and_body["airvelocity_x"],fly_trajectory_and_body["airvelocity_y"]
    fly_trajectory_and_body["airspeed"] = np.sqrt(airspeed_x**2 + airspeed_y**2)
    fly_trajectory_and_body["airspeed_angle"] = np.arctan2(airspeed_y,airspeed_x)
    
    thrust_x, thrust_y = thrust(fly_trajectory_and_body)
    fly_trajectory_and_body["thrust"] = np.sqrt(thrust_x ** 2 + thrust_y ** 2)
    fly_trajectory_and_body["thrust_angle"] = np.arctan2(thrust_y, thrust_x)

    accel_x, accel_y = _linear_acceleration(fly_trajectory_and_body)
    fly_trajectory_and_body["linear_acceleration"] = np.sqrt(accel_x ** 2 + accel_y ** 2)
    fly_trajectory_and_body["linear_acceleration_angle"] = np.arctan2(accel_y, accel_x)
    
    fly_trajectory_and_body = _transform_timestamps_to_start_at_zero(fly_trajectory_and_body)
    fly_trajectory_and_body["heading_angle"] = _tranform_short_axis_to_heading_angle(fly_trajectory_and_body['ellipse_short_angle'])
    fly_trajectory_and_body["heading_angle"] = correct_heading_jumps(fly_trajectory_and_body)
    
    fly_trajectory_and_body["angular_velocity"] = _compute_angular_velocity(fly_trajectory_and_body)
    fly_trajectory_and_body["angular_acceleration"] = _compute_angular_acceleration(fly_trajectory_and_body)
    return fly_trajectory_and_body

def _tranform_short_axis_to_heading_angle(angle):
    angle = np.where(angle < 0, angle - np.pi / 2, angle)
    angle = np.where(angle > 0, angle + np.pi / 2, angle)
    return angle


def _transform_timestamps_to_start_at_zero(fly_trajectory_and_body):
    fly_trajectory_and_body["timestamp"] = fly_trajectory_and_body["timestamp"] - fly_trajectory_and_body["timestamp"][0]
    return fly_trajectory_and_body


def _linear_acceleration(fly_trajectory_and_body):
    # calculate ground acceleration
    params = [2, 10, 10]
    dt = np.median(np.diff(fly_trajectory_and_body.timestamp))
    _, accel_x = pynumdiff.linear_model.savgoldiff(fly_trajectory_and_body.velocity_x.values, dt,
                                                                 params)
    _, accel_y = pynumdiff.linear_model.savgoldiff(fly_trajectory_and_body.velocity_y.values, dt,
                                                                 params)
    return accel_x, accel_y

def thrust(fly_trajectory_and_body, mass=0.25e-6):
    dragcoeff = mass / 0.170

    accel_x, accel_y = _linear_acceleration(fly_trajectory_and_body)

    thrust_x = mass * accel_x + dragcoeff * fly_trajectory_and_body.airvelocity_x.values  # kinematic equations
    thrust_y = mass * accel_y + dragcoeff * fly_trajectory_and_body.airvelocity_y.values


    return thrust_x,thrust_y


def _compute_angular_velocity(fly_trajectory_and_body):
# calculate ground acceleration
    params = [2, 10, 10]
    dt = np.median(np.diff(fly_trajectory_and_body.timestamp))
    _, angular_vel = pynumdiff.linear_model.savgoldiff(fly_trajectory_and_body.heading_angle, dt,
                                                                 params)
    return angular_vel

def _compute_angular_acceleration(fly_trajectory_and_body):
    params = [2, 10, 10]
    dt = np.median(np.diff(fly_trajectory_and_body.timestamp))
    _, angular_accel = pynumdiff.linear_model.savgoldiff(fly_trajectory_and_body.angular_velocity, dt,
                                                                 params)
    return angular_accel

def great_circle_distance(angle1, angle2):
    delta_theta = abs(angle2 - angle1)
    return min(delta_theta, 2 * np.pi - delta_theta)
