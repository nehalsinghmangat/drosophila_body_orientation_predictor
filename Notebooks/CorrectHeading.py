import utils
import scipy
import numpy as np
from AugmentData import *
from cvxpy.atoms.norm import norm
from cvxpy.expressions.expression import Expression

def correct_heading_jumps(traj):
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
    
    traj_add = traj.copy()

    # Heading
    angle = traj_add['heading_angle'].values # heading_angle

    # Align initial heading with course direction
    initial_window = 5
    course_direction = traj_add["airspeed_angle"]
    circ_diff_start = circular_distance(scipy.stats.circmean(course_direction[0:initial_window], low=-np.pi, high=np.pi),
                                        scipy.stats.circmean(angle[0:initial_window], low=-np.pi, high=np.pi))
    if circ_diff_start > 0.5*np.pi:
        angle = angle + np.pi * np.sign(circ_diff_start) 

    # Correct heading
    corrected_heading_angle = np.unwrap(angle, period=np.pi, discont=0.5*np.pi)  # use unwrap function to detect pi flips
    
    # Align heading
    phi_mean = scipy.stats.circmean(utils.wrapToPi(corrected_heading_angle), low=-np.pi, high=np.pi)
    psi_mean = scipy.stats.circmean(utils.wrapToPi(course_direction), low=-np.pi, high=np.pi)
    circ_diff = circular_distance(phi_mean, psi_mean)
    
    if circ_diff > 0.5*np.pi:
        corrected_heading_angle = corrected_heading_angle + np.pi * np.sign(circ_diff)
     
    corrected_heading_angle = utils.wrapToPi(corrected_heading_angle)  # wrap
    
    return corrected_heading_angle