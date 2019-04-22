import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angel):

        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1,max_lat_accel, max_steer_angel)

        kp = 0.3
        ki = 0.05
        kd = 0.0
        mn =0.0
        mx = 0.2
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.total_vehicle_mass = self.vehicle_mass + self.fuel_capacity / GAS_DENSITY

        self.last_time =rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0
            brake = 400
        elif throttle< 0.1 and vel_error < 0.0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake =abs(decel) * self.total_vehicle_mass * self.wheel_radius

        return throttle, brake, steering
