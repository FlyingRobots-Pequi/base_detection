#!/usr/bin/env python3

"""
Shutdown Client Node for Base Detection System.

This module implements a ROS2 client node that can trigger a graceful shutdown
of the coordinate processor node. It's used for safe system termination and
cleanup during the base detection process.

The node:
- Creates a client for the shutdown service
- Waits for service availability
- Sends shutdown request
- Handles service response

Dependencies:
    - ROS2
    - std_srvs
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class ShutdownClient(Node):
    """
    A ROS2 client node for triggering coordinate processor shutdown.

    This class implements a service client that can request the coordinate
    processor node to perform a graceful shutdown. It waits for service
    availability and handles the response from the shutdown request.

    Attributes:
        cli (rclpy.client.Client): ROS2 service client for shutdown requests
        req (std_srvs.srv.Trigger.Request): Service request message
        future (rclpy.task.Future): Future object for tracking service response
    """

    def __init__(self):
        """
        Initialize the ShutdownClient node.

        Sets up the service client and waits for the shutdown service to become
        available. Creates an empty trigger request message.
        """
        super().__init__('shutdown_client')
        self.cli = self.create_client(Trigger, 'coordinate_processor/shutdown')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for shutdown service...')
        self.req = Trigger.Request()

    def send_request(self):
        """
        Send shutdown request to the coordinate processor.

        Asynchronously calls the shutdown service and waits for the response.
        Logs the result of the service call, including any error messages
        if the call fails.

        Note:
            This method blocks until the service call completes or fails.
        """
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        if self.future.result() is not None:
            self.get_logger().info('Response: %s' % self.future.result().message)
        else:
            self.get_logger().error('Service call failed')


def main(args=None):
    """
    Main entry point for the shutdown client node.

    Creates the client node, sends the shutdown request, and performs cleanup.

    Args:
        args: Command line arguments (unused)
    """
    rclpy.init(args=args)
    shutdown_client = ShutdownClient()
    shutdown_client.send_request()
    shutdown_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()