import rospy
from hdas_msg.msg import motor_control  # Import the custom message
from sensor_msgs.msg import JointState
from std_msgs.msg import Header  # For the header field


class A1XController:
    def __init__(
        self,
        arm_control_topic: str = "/motion_control/control_arm",
        gripper_control_topic: str = "/motion_control/control_gripper",
        arm_state_topic: str = "/hdas/feedback_arm",
        rate_hz: float = 100,
    ):
        self.pub = rospy.Publisher(arm_control_topic, motor_control, queue_size=1)
        self.gripper_pub = rospy.Publisher(gripper_control_topic, motor_control, queue_size=1)
        self.sub = rospy.Subscriber(arm_state_topic, JointState, self.arm_state_callback)
        self.rate = rospy.Rate(rate_hz)

        self.qpos = [0.0] * 6
        self.qvel = [0.0] * 6
        self.qpos_gripper = 0.0
        self.qvel_gripper = 0.0
        self.timestamp = 0.0

        # motor control parameters
        self.q_des = None
        self.v_des = [0.0] * 6
        # self.kp = [2000, 2000, 1000, 200, 200, 200]
        # self.kd = [200.0, 500.0, 500, 200, 200, 200]
        # self.kp = [200, 200, 150, 100, 100, 100]
        # self.kd = [20.0, 30.0, 10, 10, 10, 10]
        self.kp = [140, 200, 120, 80, 80, 80]
        self.kd = [10, 30, 5, 10, 10, 10]
        self.t_ff = [0.0] * 6
        self.arm_ctrl_msg = motor_control()

        self.q_des_gripper = [-2.1]
        self.v_des_gripper = [0.0]
        self.kp_gripper = [1]
        self.kd_gripper = [0.05]
        self.t_ff_gripper = [0.0]
        self.gripper_ctrl_msg = motor_control()

    def arm_state_callback(self, msg: JointState):
        """
        Callback function to handle joint state updates.
        """
        self.qpos = msg.position[:6]
        self.qvel = msg.velocity[:6]
        self.qpos_gripper = [msg.position[6]]
        self.qvel_gripper = [msg.velocity[6]]
        if self.q_des is None:
            self.q_des = self.qpos
        self.timestamp = msg.header.stamp.to_sec()
        # print(f"Received joint state: {self.qpos}, {self.qvel} at time {self.timestamp}")
        # print(f"Gripper state: {self.qpos_gripper}, {self.qvel_gripper}")

    def publish_arm_control(self):
        """
        Publishes motor control messages to the /motion_control/control_arm topic.
        """
        if self.q_des is None:
            return

        self.arm_ctrl_msg.header = Header()
        self.arm_ctrl_msg.header.stamp = rospy.Time.now()
        self.arm_ctrl_msg.header.frame_id = "base_link"
        self.arm_ctrl_msg.kp = self.kp
        self.arm_ctrl_msg.kd = self.kd
        self.arm_ctrl_msg.t_ff = self.t_ff
        self.arm_ctrl_msg.p_des = self.q_des
        self.arm_ctrl_msg.v_des = self.v_des

        self.pub.publish(self.arm_ctrl_msg)

    def publish_gripper_control(self):
        self.gripper_ctrl_msg.header = Header()
        self.gripper_ctrl_msg.header.stamp = rospy.Time.now()
        self.gripper_ctrl_msg.header.frame_id = "gripper_link"
        self.gripper_ctrl_msg.kp = self.kp_gripper
        self.gripper_ctrl_msg.kd = self.kd_gripper
        self.gripper_ctrl_msg.t_ff = self.t_ff_gripper
        self.gripper_ctrl_msg.p_des = self.q_des_gripper
        self.gripper_ctrl_msg.v_des = self.v_des_gripper

        self.gripper_pub.publish(self.gripper_ctrl_msg)

    def run(self):
        """
        Main loop to run the controller.
        """
        while not rospy.is_shutdown():
            self.publish_motor_control()
            self.rate.sleep()
