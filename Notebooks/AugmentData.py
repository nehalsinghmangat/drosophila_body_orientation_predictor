import numpy as np
import pynumdiff


def augment_df(fly_trajectory_and_body):
    accel_x, accel_y = linear_acceleration(fly_trajectory_and_body)
    fly_trajectory_and_body["linear_acceleration"] = np.sqrt(accel_x ** 2 + accel_y ** 2)
    fly_trajectory_and_body["linear_acceleration_angle"] = np.arctan2(accel_y, accel_x)

    thrust_x, thrust_y = thrust(fly_trajectory_and_body)
    fly_trajectory_and_body["thrust"] = np.sqrt(thrust_x ** 2 + thrust_y ** 2)
    fly_trajectory_and_body["thrust_angle"] = np.arctan2(thrust_y, thrust_x)

    fly_trajectory_and_body = transform_timestamps_to_start_at_zero(fly_trajectory_and_body)
    fly_trajectory_and_body["heading_angle"] = tranform_short_axis_to_heading_angle(fly_trajectory_and_body['ellipse_short_angle'])
    return fly_trajectory_and_body

def tranform_short_axis_to_heading_angle(angle):
    angle = np.where(angle < 0, angle - np.pi / 2, angle)
    angle = np.where(angle > 0, angle + np.pi / 2, angle)
    return angle


def transform_timestamps_to_start_at_zero(fly_trajectory_and_body):
    fly_trajectory_and_body["timestamp"] = fly_trajectory_and_body["timestamp"] - fly_trajectory_and_body["timestamp"][0]
    return fly_trajectory_and_body


def linear_acceleration(fly_trajectory_and_body):
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

    accel_x, accel_y = linear_acceleration(fly_trajectory_and_body)

    thrust_x = mass * accel_x + dragcoeff * fly_trajectory_and_body.airvelocity_x.values  # kinematic equations
    thrust_y = mass * accel_y + dragcoeff * fly_trajectory_and_body.airvelocity_y.values


    return thrust_x,thrust_y

def heading_angle(fly_trajectory_and_body):
    heading_angle = tranform_short_axis_to_heading_angle(fly_trajectory_and_body['angle'].values)
    return heading_angle

def great_circle_distance_between_corrected_heading_angle(fly_trajectory_and_body):
    def II(angle):
        if angle > np.pi / 2 and angle < np.pi:
            return True
        else:
            return False

    def III(angle):
        if angle < 0 and angle < -np.pi / 2:
            return True
        else:
            return False

    def one_in_II_one_in_III(x_1, x_2):
        if (II(x_1) or II(x_2)) and (III(x_1) or III(x_2)):
            return True

    heading_angle = fly_trajectory_and_body.heading_angle.values
    great_circle_distance_between_corrected_heading_angle = []

    for index in range(0, len(heading_angle) - 1):
        x_1 = heading_angle[index]
        x_2 = heading_angle[index + 1]

        if one_in_II_one_in_III(x_1, x_2):
            great_circle_distance = 2 * np.pi - abs(x_2 - x_1)
        else:
            great_circle_distance = abs(x_2 - x_1)
        great_circle_distance_between_corrected_heading_angle.append(great_circle_distance)
    return great_circle_distance_between_corrected_heading_angle

def angular_acceleration(fly_trajectory_and_body):
    pass

def angular_velocity(fly_trajectory_and_body):
    pass