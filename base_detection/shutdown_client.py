#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class ShutdownClient(Node):
    def __init__(self):
        super().__init__('shutdown_client')
        self.cli = self.create_client(Trigger, 'coordinate_processor/shutdown')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for shutdown service...')
        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result() is not None:
            self.get_logger().info('Response: %s' % self.future.result().message)
        else:
            self.get_logger().error('Service call failed')

def main(args=None):
    rclpy.init(args=args)
    shutdown_client = ShutdownClient()
    shutdown_client.send_request()
    shutdown_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()