import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np


def plot_trajectory(fly_trajectory_and_body, every_nth=4, L=0.008):
    fig, ax = plt.subplots()
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

    ax.set_title(fly_trajectory_and_body["trajec_objid"][0])
    ax.legend()