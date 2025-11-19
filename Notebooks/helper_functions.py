import fly_plot_lib_plot as fpl
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.pyplot as plt
import cvxpy
import pandas as pd
import numpy as np
import scipy
import pynumdiff as pynd

red_cmap = mcolors.LinearSegmentedColormap.from_list('red_cmap', ['orange', 'orange'])


def plot_trajectory(ax, fly_trajectory_and_body, plot_ellipses: bool = True, every_nth=4, L=0.008, legend=True):
    ax.set_aspect('equal')

    # Adjust the limits to zoom in on the trajectory
    padding = 0.01
    ax.set_xlim(fly_trajectory_and_body['position_x'].min() - padding,
                fly_trajectory_and_body['position_x'].max() + padding)
    ax.set_ylim(fly_trajectory_and_body['position_y'].min() - padding,
                fly_trajectory_and_body['position_y'].max() + padding)

    # Plot the entire trajectory path
    ax.plot(fly_trajectory_and_body['position_x'].values,
            fly_trajectory_and_body['position_y'].values,
            color='red', label='Trajectory', linewidth=0.8,zorder=1,)

    # Optional ellipses
    if plot_ellipses:
        for ix in range(0, len(fly_trajectory_and_body), every_nth):
            ellipse = matplotlib.patches.Ellipse(
                [fly_trajectory_and_body['position_x'].iloc[ix],
                 fly_trajectory_and_body['position_y'].iloc[ix]],
                L * fly_trajectory_and_body["eccentricity"].iloc[ix],
                L,
                angle=fly_trajectory_and_body["ellipse_short_angle"].iloc[ix] * 180 / np.pi,
                color='black'
            )
            ax.add_artist(ellipse)

    # ↓↓↓ Prepare heading input for triangle plotting ↓↓↓
    position_x = fly_trajectory_and_body['position_x'].iloc[::every_nth].values
    position_y = fly_trajectory_and_body['position_y'].iloc[::every_nth].values
    heading_angle = fly_trajectory_and_body["heading_angle"].iloc[::every_nth].values
    timestamp = fly_trajectory_and_body["timestamp"].iloc[::every_nth].values  # for coloring

    # 🔺 Replace quiver heading with triangle arrows
    fpl.colorline_with_heading(
        ax=ax,
        x=position_x,
        y=position_y,
        color=timestamp,             # renamed from `c` → `color`
        orientation=heading_angle,   # renamed from `phi` → `orientation`
        nskip=0,
        size_radius=0.008,
        deg=False,
        colormap=red_cmap,
        center_point_size=0.0001,
        colornorm=None,
        show_centers=False,
        size_angle=20,
        alpha=0.8,
        edgecolor='none'
    )   

    # (Optional) other vector fields remain
    plot_scale = 0.015
    thrust_scale = 5000

    airvelocity_x = fly_trajectory_and_body['airvelocity_x'].iloc[::every_nth].values
    airvelocity_y = fly_trajectory_and_body['airvelocity_y'].iloc[::every_nth].values

    thrust_angle = fly_trajectory_and_body['thrust_angle'].iloc[::every_nth].values
    thrust_magnitude = fly_trajectory_and_body['thrust'].iloc[::every_nth].values
    thrust_x = np.cos(thrust_angle) * thrust_magnitude
    thrust_y = np.sin(thrust_angle) * thrust_magnitude

    ax.quiver(position_x, position_y, airvelocity_x * plot_scale, airvelocity_y * plot_scale,
              color="blue", angles='xy', scale_units='xy', scale=1, width=0.0045, label="Air velocity")

    ax.quiver(position_x, position_y, thrust_x * thrust_scale, thrust_y * thrust_scale,
              color="green", angles='xy', scale_units='xy', scale=1, width=0.0045, label="Thrust")

    ax.set_title(fly_trajectory_and_body["trajec_objid"].iloc[0])
    if legend:
        ax.legend()

def naive_heading_correction(fly_trajectory_and_body):
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
    angle = traj_add["heading_angle"] # heading_angle

    # Align initial heading with thrust angle
    initial_window = 5
    course_direction = traj_add["groundspeed_angle"]
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
    fly_trajectory_and_body["heading_angle"] = corrected_heading_angle
    return fly_trajectory_and_body

def convex_opt_heading_correction(fly_trajectory_and_body: pd.DataFrame):
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
    thrust_angle = fly_trajectory_and_body.thrust_angle.values

    diff_btwn_heading_and_thrust = angle_difference(heading_angle, thrust_angle)
    avg_diff_btwn_heading_and_thrust = np.average(diff_btwn_heading_and_thrust)
    
    # terms
    L1 = cvxpy.tv(heading_angle + np.pi * k)
    L3 = cvxpy.norm1(avg_diff_btwn_heading_and_thrust + np.pi * k)  # theta + np.pi*k - thrust_angle

    # coefficients
    alpha1 = 1
    alpha3 = 5

    L = alpha1 * L1 + alpha3 * L3

    # solve the optimization
    constraints = [-1 <= k, k <= 1]
    obj = cvxpy.Minimize(L)
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver='MOSEK')

    # output
    corrected_heading_angle = wrap_angle(heading_angle + k.value * np.pi)
    fly_trajectory_and_body["heading_angle"] = corrected_heading_angle
    return fly_trajectory_and_body

def smooth_trajectory(trajectory: pd.DataFrame,savgol_params: list=[1,5,5]) -> pd.DataFrame:
    def unwrap_angle(z, correction_window_for_2pi=100, n_range=2, plot=False):
        if 0: # option one
            zs = []
            for n in range(-1*n_range, n_range):
                zs.append(z+n*np.pi*2)
            zs = np.vstack(zs)

            smooth_zs = np.array(z[0:2])

            for i in range(2, len(z)):
                first_ix = np.max([0, i-correction_window_for_2pi])
                last_ix = i
                error = np.abs(zs[:,i] - np.mean(smooth_zs[first_ix:last_ix])) 
                smooth_zs = np.hstack(( smooth_zs, [zs[:,i][np.argmin(error)]] ))

            if plot:
                for r in range(zs.shape[0]):
                    plt.plot(zs[r,:], '.', markersize=1)
                plt.plot(smooth_zs, '.', color='black', markersize=1)
            
        else: # option two, automatically scales n_range to most recent value, and maybe faster
            smooth_zs = np.array(z[0:2])
            for i in range(2, len(z)):
                first_ix = np.max([0, i-correction_window_for_2pi])
                last_ix = i
                
                nbase = np.round( (smooth_zs[-1] - z[i])/(2*np.pi) )
                
                candidates = []
                for n in range(-1*n_range, n_range):
                    candidates.append(n*2*np.pi+nbase*2*np.pi+z[i])
                error = np.abs(candidates - np.mean(smooth_zs[first_ix:last_ix])) 
                smooth_zs = np.hstack(( smooth_zs, [candidates[np.argmin(error)]] ))
            if plot:
                plt.plot(smooth_zs, '.', color='black', markersize=1)
        return smooth_zs
    trajectory_copy = trajectory.copy()
    heading_angle = np.array(trajectory_copy["heading_angle"])
    smooth_heading_angle,_ = pynd.savgoldiff(unwrap_angle(heading_angle),dt=0.01,params=savgol_params)
    trajectory_copy["heading_angle"] = smooth_heading_angle
    return trajectory_copy


def custom_density_plots(axes: list[plt.Axes], 
                         training_and_testing_X_data: list[np.ndarray], 
                         training_and_testing_y_data: list[np.ndarray], 
                         best_estimator, 
                         cmap,
                         titles: list[str] = ["Training Data", "Testing Data"]) -> None:
    
    for axis, Xdata, ydata, title in zip(axes, training_and_testing_X_data, training_and_testing_y_data, titles):
        # Predict using the best estimator
        Y_predict = best_estimator.predict(Xdata, batch_size=4096)

        # Calculate the predicted and true headings using np.arctan2
        zeta_predict = np.arctan2(Y_predict[:, 1], Y_predict[:, 0])
        zeta_true = np.arctan2(ydata.values[:, 1], ydata.values[:, 0])

        # Wrap the angles to [0, 2*pi)
        zeta_predict = (zeta_predict + 2 * np.pi) % (2 * np.pi)
        zeta_true = (zeta_true + 2 * np.pi) % (2 * np.pi)

        # Create 2D histogram with density normalization and log scale
        h = axis.hist2d(zeta_true, zeta_predict, bins=(128, 128), density=True, cmap=cmap,norm=mcolors.LogNorm(clip=True))
        cbar = plt.colorbar(h[3], ax=axis)
        cbar.set_label('Density (log scale)')

        # Grid and labels
        axis.set_ylim(0 - 0.1, 2 * np.pi + 0.1)
        axis.set_xlim(0 - 0.1, 2 * np.pi + 0.1)
        axis.set_ylabel('Predicted Heading (rad)')
        axis.set_xlabel('True Heading (rad)')
        axis.set_title(title)

blue_cmap = mcolors.LinearSegmentedColormap.from_list('blue_cmap', ['blue', 'blue'])
red_cmap = mcolors.LinearSegmentedColormap.from_list('red_cmap', ['red', 'red'])

from scipy.ndimage import gaussian_filter1d

def augment_with_time_delay_embedding(fly_traj_list: list[pd.DataFrame],**kwargs):
    def collect_offset_rows(df, aug_column_names=None, keep_column_names=None, w=1, direction='backward'):
        """ Takes a pandas data frame with n rows, list of columns names, and a window size w.
            Then creates an augmented data frame that collects prior or future rows (in window)
            and stacks them as new columns. The augmented data frame will be size (n - w - 1) as the first/last
            w rows do not have enough data before/after them.
            
            CEM addition: Assumes dataframe contains van Breugel wind tunnel data with single wind direction for
            all data. Assigns randomly generated rotation to all angles in each trajectory to simulate multiple
            wind directions. Used to preprocess training dataset for `model-CEM_all-angle-rotate.keras`.

            Inputs
                df: pandas data frame
                aug_column_names: names of the columns to augment
                keep_column_names: names of the columns to keep, but not augment
                w: lookback window size (# of rows)
                direction: get the rows from behind ('backward') or front ('forward')

            Outputs
                df_aug: augmented pandas data frame.
                        new columns are named: old_name_0, old_name_1, ... , old_name_w-1
        """

        df = df.reset_index(drop=True)

        # Default for testing
        if df is None:
            df = np.atleast_2d(np.arange(0, 11, 1, dtype=np.double)).T
            df = np.matlib.repmat(df, 1, 4)
            df = pd.DataFrame(df, columns=['a', 'b', 'c', 'd'])
            aug_column_names = ['a', 'b']
        else:  # use the input  values
            # Default is all columns
            if aug_column_names is None:
                aug_column_names = df.columns

        # Make new column names & dictionary to store data
        new_column_names = {}
        df_aug_dict = {}
        for a in aug_column_names:
            new_column_names[a] = []
            df_aug_dict[a] = []

        for a in aug_column_names:  # each augmented column
            for k in range(w):  # each point in lookback window
                new_column_names[a].append(a + '_' + str(k))

        # Augment data
        n_row = df.shape[0]  # # of rows
        n_row_train = n_row - w + 1  # # of rows in augmented data
        for a in aug_column_names:  # each column to augment
            data = df.loc[:, [a]]  # data to augment
            data = np.asmatrix(data)  # as numpy matrix
            df_aug_dict[a] = np.nan * np.ones((n_row_train, len(new_column_names[a])))  # new augmented data matrix

            # Put augmented data in new column, for each column to augment
            for i in range(len(new_column_names[a])):  # each column to augment
                if direction == 'backward':
                    # Start index, starts at the lookback window size & slides up by 1 for each point in window
                    startI = w - 1 - i

                    # End index, starts at end of the matrix &  & slides up by 1 for each point in window
                    endI = n_row - i  # end index, starts at end of matrix &

                elif direction == 'forward':
                    # Start index, starts at the beginning of matrix & slides up down by 1 for each point in window
                    startI = i

                    # End index, starts at end of the matrix minus the window size
                    # & slides down by 1 for each point in window
                    endI = n_row - w + 1 + i  # end index, starts at end of matrix &

                else:
                    raise Exception("direction must be 'forward' or 'backward'")

                # Put augmented data in new column
                df_aug_dict[a][:, i] = np.squeeze(data[startI:endI, :])

            # Convert data to pandas data frame & set new column names
            df_aug_dict[a] = pd.DataFrame(df_aug_dict[a], columns=new_column_names[a])

        # Combine augmented column data
        df_aug = pd.concat(list(df_aug_dict.values()), axis=1)

        # Add non-augmented data, if specified
        if keep_column_names is not None:
            for c in keep_column_names:
                if direction == 'backward':
                    startI = w - 1
                    endI = n_row
                elif direction == 'forward':
                    startI = 0
                    endI = n_row - w
                else:
                    raise Exception("direction must be 'forward' or 'backward'")

                keep = df.loc[startI:endI, [c]].reset_index(drop=True)
                df_aug = pd.concat([df_aug, keep], axis=1)

        return df_aug
    time_window = kwargs["time_window"]
    input_names = kwargs["input_names"]
    output_names = kwargs["output_names"]
    direction = kwargs["direction"]
    traj_augment_list = []
    
    for i in range(len(fly_traj_list)):
        traj = fly_traj_list[i].copy()
        traj['heading_angle_x'] = np.cos(traj['heading_angle'])                     #(CEM addition!)
        traj['heading_angle_y'] = np.sin(traj['heading_angle'])                     #(CEM addition!)
        traj_augment = collect_offset_rows(traj,
                                           aug_column_names=input_names,
                                           keep_column_names=output_names,
                                           w=time_window,
                                           direction=direction)

        traj_augment_list.append(traj_augment)

    traj_augment_all = pd.concat(traj_augment_list, ignore_index=True)

    return np.round(traj_augment_all, 4)

def plot_trajectory_with_predicted_heading(trajectory: pd.DataFrame, axis: plt.Axes, n_input: int,best_estimator,nskip: int=0,arrow_size=None,include_id=False,plt_show=False,smooth=False,**kwargs):
    def predict_heading_from_fly_trajectory(df: pd.DataFrame, n_input, augment_with_time_delay_embedding: callable, estimator: callable, **kwargs):
        augmented_df = augment_with_time_delay_embedding([df],**kwargs)
        augmented_df = augmented_df.iloc[:, 0:n_input]
        heading_components= estimator.predict(augmented_df)
        if smooth:
            # Gaussian smoothing parameters
            sigma = 2  # Standard deviation for the Gaussian filter, adjust as needed
            heading_components= gaussian_filter1d(estimator.predict(augmented_df),sigma=sigma,axis=0)
        heading_angle_predicted = np.arctan2(heading_components[:,1],heading_components[:,0])
        number_of_beginning_time_steps_deleted = len(df["position_x"]) - len(heading_angle_predicted) # obviously, this will not be the beginning time steps if you change the time augmentation to "forward"; similarly, a "center" time augmentation would delete things on both ends

        # Step 1: Extract the first value
        first_value = heading_angle_predicted[0]

        # Step 2: Create an array with the first value repeated 3 times
        prepend_values = np.array([first_value] * number_of_beginning_time_steps_deleted)

        # Step 3: Concatenate the new array with the original array
        heading_angle_predicted_arr = np.concatenate([prepend_values, heading_angle_predicted])
        return heading_angle_predicted_arr
    
    def plot_trajectory(xpos, ypos, phi, color, ax=None, size_radius=None, nskip=0,
                colormap='bone_r', colornorm=None, edgecolor='none', reverse=False,alpha=0.7):
        if color is None:
                color = phi

        color = np.array(color)

            #Set size radius
        xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
        if size_radius is None:  # auto set
            xymean = 0.21 * xymean
            if xymean < 0.0001:
                sz = np.array(0.01)
            else:
                sz = np.hstack((xymean, 1))
            size_radius = sz[sz > 0][0]
        else:
            if isinstance(size_radius, list):  # scale default by scalar in list
                xymean = size_radius[0] * xymean
                sz = np.hstack((xymean, 1))
                size_radius = sz[sz > 0][0]
            else:  # use directly
                size_radius = size_radius

        if colornorm is None:
            colornorm = [np.min(color), np.max(color)]

        if reverse:
            xpos = np.flip(xpos, axis=0)
            ypos = np.flip(ypos, axis=0)
            phi = np.flip(phi, axis=0)
            color = np.flip(color, axis=0)

        fpl.colorline_with_heading(ax, np.flip(xpos), np.flip(ypos), np.flip(color, axis=0), np.flip(phi),
                                    nskip=nskip,
                                    size_radius=size_radius,
                                    deg=False,
                                    colormap=colormap,
                                    center_point_size=0.0001,
                                    colornorm=colornorm,
                                    show_centers=False,
                                    size_angle=20,
                                    alpha=alpha,
                                    edgecolor=edgecolor)

        ax.set_aspect('equal')
        
        # Define fixed minimum plot size
        min_size = 0.1  # Adjust this value as needed

        xrange = xpos.max() - xpos.min()
        xrange = np.max([xrange, min_size])
        yrange = ypos.max() - ypos.min()
        yrange = np.max([yrange, min_size])

        ax.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
        ax.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)

        if include_id:
            ax.set_title(trajectory['trajec_objid'].iloc[0])
    heading_angle_predicted = predict_heading_from_fly_trajectory(trajectory,n_input,augment_with_time_delay_embedding,best_estimator,**kwargs)

    # Plot the predicted heading trajectory first so that it is under the actual trajectory
    plot_trajectory(trajectory.position_x.values,
                    trajectory.position_y.values,
                    heading_angle_predicted, 
                    trajectory.timestamp.values,
                    ax=axis,
                    size_radius=arrow_size,
                    nskip=nskip,
                    colormap=blue_cmap,  
                    colornorm=None,
                    edgecolor='none',
                    reverse=False,
                    alpha=0.7) # Different shade for distinction
    
    # Plot actual trajectory
    plot_trajectory(trajectory.position_x.values,
                    trajectory.position_y.values,
                    trajectory.heading_angle.values,
                    trajectory.timestamp.values,
                    ax=axis,
                    size_radius=arrow_size,
                    nskip=nskip,
                    colormap=red_cmap,
                    colornorm=None,
                    edgecolor="black",
                    reverse=False,
                    alpha=0.3)

def plot_fly_inputs_stacked(df, axes=None):
    import matplotlib.pyplot as plt

    # Columns you want, one axis each (top to bottom)
    input_cols = [
        'groundspeed',
        'groundspeed_angle',
        'airspeed',
        'airspeed_angle',
        'thrust',
        'thrust_angle',
    ]

    if axes is None:
        fig, axes = plt.subplots(
            len(input_cols), 1,
            figsize=(10, 8),
            dpi=150,
            sharex=True
        )

    # Make sure axes is an array-like
    if not isinstance(axes, (list, tuple)):
        # if a single axis was passed, wrap it
        axes = [axes]

    # Plot each column on its own axis
    for ax, col in zip(axes, input_cols):
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col)
            ax.set_ylabel(col)
            ax.grid(True)
        else:
            ax.set_visible(False)  # hide if column missing

    # Only label x-axis on the bottom plot
    axes[-1].set_xlabel("Time (s)")

    return axes