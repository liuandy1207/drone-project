#approximate as rectangular surfacees
#x: .15887 by .07059
#y: .09562 by .04
#z: 0.15819 by 0.035
#Assume changes in surface area due to drone movement/tilt are negligble
#or a rectangle facing the flow, assume  Cd ​ ≈1.28.
rho = 1.225
C_d = 1.28

def drag(v_x, v_y, v_z):
    a_x = .15887 * .07059
    d_x = rho * (v_x)^2 * C_d * a_x * 1/2

    a_y = .09562 * .04
    d_y = rho * (v_y)^2 * C_d * a_y * 1/2

    a_z = 0.15819 * 0.035
    d_z = rho * (v_z)^2 * C_d * a_z * 1/2

    return d_x, d_y, d_z
