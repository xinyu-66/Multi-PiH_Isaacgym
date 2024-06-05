import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import select
import sys

class PublisherNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.pub = self.create_publisher(String, "channel", 10)
        self.msg = String()
        # self.timer = self.create_timer(1, self.timer_callback)


    def manage_input(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            cmd = sys.stdin.read(1)
            if cmd == 'q':
                print("q_keyboard")
            
            self.msg.data = cmd
            self.pub.publish(self.msg)
            self.get_logger().info("Publishing: '%s', with" % (self.msg.data))

    # def timer_callback(self):
    #     msg = String()
    #     msg.data = "xinde DSB CNM"
    #     self.pub.publish(msg)
    #     self.get_logger().info("Publishing: '%s' " % msg.data)


def run():
    node = PublisherNode("topic_channel_pub")
    # msg = String()
    # msg.data = "xinde DSB CNM"
    flag = 0

    while rclpy.ok():

        rclpy.spin_once(node, timeout_sec=0.5)
        # print("good", flag)

        # print("spining", flag)
        node.manage_input()
        # for i in range(1,10):
            

        # print("spining)in", flag)
        # print("spining)out", flag)
        flag += 1
    
    # rclpy.spin_once(node, timeout_sec=10)
    node.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    run()

    
    

    
