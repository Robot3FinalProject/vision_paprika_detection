import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class StatusSubscriberNode(Node):
    def __init__(self):
        super().__init__('status_subscriber_node')
        
        # [ROS 2 Subscriber 생성]
        # 비전 노드가 퍼블리시하는 '/vision/box_status' 토픽을 구독(listen)합니다.
        self.subscription = self.create_subscription(
            String,
            '/vision/box_status',
            self.listener_callback,
            10
        )
        self.get_logger().info("📥 Status Subscriber Node has been started. Waiting for vision data...")

    def listener_callback(self, msg):
        # 토픽 메시지가 수신될 때마다 호출되는 콜백 함수
        self.get_logger().info(f"📦 [BOX STATUS RECEIVED] -> {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    subscriber_node = StatusSubscriberNode()
    
    try:
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber_node.destroy_node()
        rclpy.shutdown()
        print("✅ Subscriber program terminated safely.")


if __name__ == '__main__':
    main()