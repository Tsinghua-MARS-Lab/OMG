from __future__ import annotations

import argparse
import struct
import time


KEY_A = 8
KEY_B = 9
DEFAULT_ACTION_TOPIC = "/humanoid/action"
DEFAULT_LOWSTATE_TOPIC = "/lowstate"
DEFAULT_ROBOT_STATE_TOPIC = "/robot_state"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive HoloMotion policy_node_29dof in ROS dry-run mode. The driver publishes "
            "fake /lowstate and /robot_state messages, simulates A then B button presses, "
            "and counts /humanoid/action messages. It never publishes /lowcmd."
        )
    )
    parser.add_argument("--duration", type=float, default=12.0, help="Total driver runtime in seconds.")
    parser.add_argument("--rate", type=float, default=50.0, help="Fake lowstate publish rate.")
    parser.add_argument("--a-start", type=float, default=1.0, help="Seconds before pressing A.")
    parser.add_argument("--a-duration", type=float, default=0.18, help="A button hold duration in seconds.")
    parser.add_argument("--b-start", type=float, default=2.5, help="Seconds before pressing B.")
    parser.add_argument("--b-duration", type=float, default=0.18, help="B button hold duration in seconds.")
    parser.add_argument("--lowstate-topic", default=DEFAULT_LOWSTATE_TOPIC)
    parser.add_argument("--robot-state-topic", default=DEFAULT_ROBOT_STATE_TOPIC)
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC)
    return parser.parse_args()


def _pack_remote_button(key_bit: int | None) -> list[int]:
    remote = bytearray(40)
    if key_bit is not None:
        remote[2:4] = struct.pack("<H", 1 << int(key_bit))
    return list(remote)


def main() -> None:
    args = _parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.rate <= 0.0:
        raise ValueError("--rate must be positive")
    if args.a_duration < 0.0 or args.b_duration < 0.0:
        raise ValueError("button durations must be non-negative")

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String
    from unitree_hg.msg import LowState

    class DryRunDriver(Node):
        def __init__(self) -> None:
            super().__init__("omg_policy_node_smoke_driver")
            self.low_pub = self.create_publisher(LowState, args.lowstate_topic, 10)
            self.state_pub = self.create_publisher(String, args.robot_state_topic, 10)
            self.action_count = 0
            self.first_action_s: float | None = None
            self.last_action_head: list[float] | None = None
            self.started_at = time.time()
            self.create_subscription(Float32MultiArray, args.action_topic, self._on_action, 10)
            self.create_timer(1.0 / float(args.rate), self._tick)

        def _on_action(self, msg: Float32MultiArray) -> None:
            self.action_count += 1
            if self.first_action_s is None:
                self.first_action_s = time.time() - self.started_at
            self.last_action_head = [float(x) for x in msg.data[:5]]

        def _fake_lowstate(self, key_bit: int | None) -> LowState:
            msg = LowState()
            msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
            msg.imu_state.gyroscope = [0.0, 0.0, 0.0]
            msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
            msg.wireless_remote = _pack_remote_button(key_bit)
            return msg

        def _tick(self) -> None:
            elapsed = time.time() - self.started_at
            state = String()
            state.data = "MOVE_TO_DEFAULT"
            self.state_pub.publish(state)

            key_bit = None
            if args.a_start <= elapsed < args.a_start + args.a_duration:
                key_bit = KEY_A
            elif args.b_start <= elapsed < args.b_start + args.b_duration:
                key_bit = KEY_B
            self.low_pub.publish(self._fake_lowstate(key_bit))

    rclpy.init()
    driver = DryRunDriver()
    try:
        deadline = time.time() + float(args.duration)
        while time.time() < deadline:
            rclpy.spin_once(driver, timeout_sec=0.05)
    finally:
        print(
            "policy_action_count="
            f"{driver.action_count} first_action_s={driver.first_action_s} "
            f"last_action_head={driver.last_action_head}",
            flush=True,
        )
        driver.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
