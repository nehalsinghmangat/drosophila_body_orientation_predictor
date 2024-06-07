import cvxpy
from AugmentData import *
from cvxpy.atoms.norm import norm
from cvxpy.expressions.expression import Expression

def correct_heading_jumps(fly_trajectory_and_body):
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

    thrust_x, thrust_y = thrust(fly_trajectory_and_body)
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
    fly_trajectory_and_body["heading_angle"] = corrected_heading_angle
    return fly_trajectory_and_body