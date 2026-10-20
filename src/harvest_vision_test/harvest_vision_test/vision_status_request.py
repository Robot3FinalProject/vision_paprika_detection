#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class VisionStatusClient(Node):
    def __init__(self):
        super().__init__('vision_status_request_client')
        self.cli = self.create_client(Trigger, 'get_latest_inserted_object')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
            
        self.req = Trigger.Request()

    def send_request(self):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main():
    rclpy.init()
    client = VisionStatusClient()
    response = client.send_request()
    
    if response is not None:
        print("\n[ROS 2 Client Request Result]")
        print(f"  - Success : {response.success}")
        print(f"  - Message : {response.message}")
        
        # 강화학습 보상(Reward) 계산용 데이터 핸들링 예시
        if response.success:
            # 성공 시: response.message에 해당하는 객체 색상(Red/Yellow) 기반 보상 부여
            reward_feedback = {"status": "SUCCESS", "object": response.message}
        else:
            # 실패 시: 페널티 부여
            reward_feedback = {"status": "FAIL", "object": None}
    else:
        print("❌ Service call failed.")
        reward_feedback = {"status": "ERROR", "object": None}
        
    client.destroy_node()
    rclpy.shutdown()
    return reward_feedback

if __name__ == '__main__':
    main()