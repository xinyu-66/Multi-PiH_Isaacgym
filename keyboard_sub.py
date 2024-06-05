import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__("subscriber_node")
        self.sub = self.create_subscription(
            String, 
            "keyboard_input", 
            self.sub_callback, 
            10)

    def sub_callback(self, msg):
        self.get_logger().info("Heard: %s " % msg.data)

# def main(args=None):
#     rclpy.init(args=args)

#     node = SubscriberNode("keyboard_input")
#     # while rclpy.ok():

    
#     rclpy.spin(node)

#     node.destroy_node()
#     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    node = SubscriberNode()

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()