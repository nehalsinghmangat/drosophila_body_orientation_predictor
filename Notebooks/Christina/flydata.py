import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import scipy
import pynumdiff
import cvxpy
class compute:

    def angular_acceleration(fly_df: pd.DataFrame, name_of_heading_field:str, name_of_time_field: str):
        params = [2, 10, 10]
        dt = np.median(np.diff(fly_df[name_of_time_field]))
        _, angular_accel = pynumdiff.linear_model.savgoldiff(compute.angular_velocity(fly_df,name_of_heading_field,name_of_time_field), dt,params)
        return angular_accel
    
    def angular_velocity(fly_df: pd.DataFrame, name_of_heading_field:str, name_of_time_field: str):
        params = [2, 10, 10]
        dt = np.median(np.diff(fly_df[name_of_time_field]))
        _, angular_vel = pynumdiff.linear_model.savgoldiff(fly_df[name_of_heading_field], dt,
                                                                    params)
        return angular_vel
    

    def heading_angle_corrected(fly_trajectory_and_body: pd.DataFrame, name_of_heading_field:str, name_of_airspeed_field: str):
        """
        The heading values randomly jump 180 degrees. This function corrects for such jumps.

        Parameters:
        - fly_data (pd.DataFrame): initial df of fly data

        Returns:
        - corrected_fly_data (pd.DataFrame): fly data with corrected heading angles
        """

        def circular_distance(angle1, angle2):
            """
            Calculate the shortest distance between two angles on a 2D circle.
            
            Parameters:
            angle1, angle2: Angles in radians.
            
            Returns:
            float: Shortest distance between the two angles on the circle.
            """
            # Normalize angles to range [0, 2π)
            angle1 = angle1 % (2 * np.pi)
            angle2 = angle2 % (2 * np.pi)
            
            # Calculate the direct distance and the wrapped-around distance
            direct_distance = np.abs(angle1 - angle2)
            wrapped_distance = 2 * np.pi - direct_distance
            
            # Return the shorter of the two distances
            return min(direct_distance, wrapped_distance)
        
        def wrapToPi(rad):
            rad_wrap = np.copy(rad)
            q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
            rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
            return rad_wrap
        
        traj_add = fly_trajectory_and_body.copy()

        # Heading
        angle = traj_add[name_of_heading_field] # heading_angle

        # Align initial heading with course direction
        initial_window = 5
        course_direction = traj_add[name_of_airspeed_field]
        circ_diff_start = circular_distance(scipy.stats.circmean(course_direction[0:initial_window], low=-np.pi, high=np.pi),
                                            scipy.stats.circmean(angle[0:initial_window], low=-np.pi, high=np.pi))
        if circ_diff_start > 0.5*np.pi:
            angle = angle + np.pi * np.sign(circ_diff_start) 

        # Correct heading
        corrected_heading_angle = np.unwrap(angle, period=np.pi, discont=0.5*np.pi)  # use unwrap function to detect pi flips
        
        # Align heading
        phi_mean = scipy.stats.circmean(wrapToPi(corrected_heading_angle), low=-np.pi, high=np.pi)
        psi_mean = scipy.stats.circmean(wrapToPi(course_direction), low=-np.pi, high=np.pi)
        circ_diff = circular_distance(phi_mean, psi_mean)
        
        if circ_diff > 0.5*np.pi:
            corrected_heading_angle = corrected_heading_angle + np.pi * np.sign(circ_diff)
        
        corrected_heading_angle = wrapToPi(corrected_heading_angle)  # wrap
        
        return corrected_heading_angle
    def heading_angle_convex_opt(fly_trajectory_and_body: pd.DataFrame):
        """
        The heading values randomly jump 180 degrees. This function corrects for such jumps.

        Parameters:
        - fly_data (pd.DataFrame): initial df of fly data

        Returns:
        - corrected_fly_data (pd.DataFrame): fly data with corrected heading angles
        """

        def angle_difference(alpha, beta):
            a = alpha - beta
            a = (a + np.pi) % (np.pi * 2) - np.pi
            return a

        def wrap_angle(a):
            return np.arctan2(np.sin(a), np.cos(a))
            
        k = cvxpy.Variable(len(fly_trajectory_and_body), integer=True)

        heading_angle = fly_trajectory_and_body.heading_angle.values

        thrust_x, thrust_y = compute.thrust(fly_trajectory_and_body)
        thrust_angle = np.arctan2(thrust_y, thrust_x)

        diff_btwn_heading_and_thrust = angle_difference(heading_angle, thrust_angle)
        avg_diff_btwn_heading_and_thrust = np.average(diff_btwn_heading_and_thrust)
        
        # terms
        L1 = cvxpy.tv(heading_angle + np.pi * k)
        L3 = cvxpy.norm1(avg_diff_btwn_heading_and_thrust + np.pi * k)  # theta + np.pi*k - thrust_angle

        # coefficients
        alpha1 = 1
        alpha3 = 1

        L = alpha1 * L1 + alpha3 * L3

        # solve the optimization
        constraints = [-1 <= k, k <= 1]
        obj = cvxpy.Minimize(L)
        prob = cvxpy.Problem(obj, constraints)
        prob.solve(solver='MOSEK')

        # output
        corrected_heading_angle = wrap_angle(heading_angle + k.value * np.pi)
        return corrected_heading_angle
    # derive heading angles from the short axis ellipse angles in the trajectory dataset
    def heading_angle_from_ellipse(angle):
        angle = np.where(angle < 0, angle - np.pi / 2, angle)
        angle = np.where(angle > 0, angle + np.pi / 2, angle)
        return angle
    
    def linear_acceleration(fly_trajectory_and_body: pd.DataFrame):
        params = [2, 10, 10]
        dt = np.median(np.diff(fly_trajectory_and_body.timestamp))
        _, accel_x = pynumdiff.linear_model.savgoldiff(fly_trajectory_and_body.velocity_x.values, dt,
                                                                    params)
        _, accel_y = pynumdiff.linear_model.savgoldiff(fly_trajectory_and_body.velocity_y.values, dt,
                                                                    params)
        return accel_x, accel_y

    def thrust(fly_trajectory_and_body: pd.DataFrame, mass=0.25e-6):
        dragcoeff = mass / 0.170

        accel_x, accel_y = compute.linear_acceleration(fly_trajectory_and_body)

        thrust_x = mass * accel_x + dragcoeff * fly_trajectory_and_body.airvelocity_x.values  # kinematic equations
        thrust_y = mass * accel_y + dragcoeff * fly_trajectory_and_body.airvelocity_y.values


        return thrust_x,thrust_y

class plot:
    def plot_trajectory(ax,fly_trajectory_and_body, every_nth=4, L=0.008):
        ax.set_aspect('equal')

        # Adjust the limits to zoom in on the trajectory
        padding = 0.01
        ax.set_xlim(fly_trajectory_and_body['position_x'].min() - padding,
                    fly_trajectory_and_body['position_x'].max() + padding)
        ax.set_ylim(fly_trajectory_and_body['position_y'].min() - padding,
                    fly_trajectory_and_body['position_y'].max() + padding)

        # Plot the entire trajectory
        ax.plot(fly_trajectory_and_body['position_x'].values, fly_trajectory_and_body['position_y'].values, color='red',
                label='Trajectory',zorder=1)

        # Plot the ellipses
        for ix in range(0, len(fly_trajectory_and_body), every_nth):
            ellipse = matplotlib.patches.Ellipse([fly_trajectory_and_body['position_x'].iloc[ix],
                                                fly_trajectory_and_body['position_y'].iloc[ix]],
                                                L * fly_trajectory_and_body["eccentricity"].iloc[ix],
                                                L,
                                                angle=fly_trajectory_and_body["ellipse_short_angle"].iloc[ix] * 180 / np.pi,
                                                color='black')
            ax.add_artist(ellipse)

        # Extract every n th for plotting convenience
        position_x = fly_trajectory_and_body['position_x'].iloc[::every_nth].values
        position_y = fly_trajectory_and_body['position_y'].iloc[::every_nth].values

        heading_angle = fly_trajectory_and_body["heading_angle"].iloc[::every_nth].values
        heading_x = np.cos(heading_angle)
        heading_y = np.sin(heading_angle)

        airvelocity_x = fly_trajectory_and_body['airvelocity_x'].iloc[::every_nth].values
        airvelocity_y = fly_trajectory_and_body['airvelocity_y'].iloc[::every_nth].values

        linear_acceleration_angle = fly_trajectory_and_body['linear_acceleration_angle'].iloc[::every_nth].values
        linear_acceleration_magnitude = fly_trajectory_and_body['linear_acceleration'].iloc[::every_nth].values
        linear_acceleration_x = np.cos(linear_acceleration_angle) * linear_acceleration_magnitude
        linear_acceleration_y = np.sin(linear_acceleration_angle) * linear_acceleration_magnitude

        thrust_angle = fly_trajectory_and_body['thrust_angle'].iloc[::every_nth].values
        thrust_magnitude = fly_trajectory_and_body['thrust'].iloc[::every_nth].values
        thrust_x = np.cos(thrust_angle) * thrust_magnitude
        thrust_y = np.sin(thrust_angle) * thrust_magnitude

        plot_scale = 0.015
        thrust_scale = 10000

        # Plot the heading vectors
        ax.quiver(position_x, position_y, heading_x*plot_scale, heading_y*plot_scale, color="darkorange", angles='xy',
                scale_units='xy', scale=1, width=0.0025, label="Heading")

        # Plot the air velocity vectors
        ax.quiver(position_x, position_y, airvelocity_x*plot_scale, airvelocity_y*plot_scale, color="blue", angles='xy',
                scale_units='xy', scale=1, width=0.0025, label="Air velocity")

        # Plot the thrust angle vectors
        ax.quiver(position_x, position_y, thrust_x*thrust_scale, thrust_y*thrust_scale, color="green", angles='xy',
                scale_units='xy', scale=1, width=0.0025, label="Thrust")

        ax.set_title(fly_trajectory_and_body["trajec_objid"].iloc[0])
        ax.legend()







