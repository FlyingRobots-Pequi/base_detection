#!/usr/bin/env python3
"""
Script para publicar posições das bases diretamente no tópico /unique_positions.

Este script simula que o sistema de detecção já detectou as 5 bases,
permitindo testar o algoritmo de pouso do fase1.py sem precisar 
rodar todo o sistema de detecção de bases.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import time

class FakeBasePublisher(Node):
    def __init__(self):
        super().__init__('fake_base_publisher')
        
        # Publisher para unique_positions (mesmo tópico que o coordinate_processor usa)
        self.unique_positions_pub = self.create_publisher(
            PoseArray,
            '/unique_positions',
            10
        )
        
        # Posições ground truth das 5 bases (extraídas do sistema)
        self.base_positions = [
            (3.26, -0.29, 0.0),   # BASE_1
            (3.29, -2.37, 0.0),   # BASE_2  
            (0.41, -3.46, 0.0),   # BASE_3
            (1.60, -5.40, 0.0),   # BASE_4
            (4.89, -5.40, 0.0),   # BASE_5
            (4.73, -3.46, 0.0)    # BASE_6
        ]
        
        self.get_logger().info(" Fake Base Publisher iniciado!")
        self.get_logger().info(" Preparando para publicar 5 bases detectadas...")
        
        # Aguarda um pouco e depois publica as bases
        self.create_timer(3.0, self.publish_fake_bases)

    def publish_fake_bases(self):
        """Publica as 5 bases como se tivessem sido detectadas"""
        
        # Cria PoseArray com as 5 bases
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        
        self.get_logger().info(" Publicando 5 bases detectadas para o fase1.py...")
        self.get_logger().info("=" * 60)
        
        for i, (x, y, z) in enumerate(self.base_positions, 1):
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0  # Quaternion padrão
            
            pose_array.poses.append(pose)
            
            distance = (x**2 + y**2)**0.5
            self.get_logger().info(f" Base {i}: ({x:6.3f}, {y:6.3f}, {z:6.3f}) - dist: {distance:.3f}m")
        
        # Publica as bases
        self.unique_positions_pub.publish(pose_array)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"   Publicadas {len(self.base_positions)} bases no tópico /unique_positions")
        self.get_logger().info("   O fase1.py agora deve receber as bases e iniciar a missão de pouso!")
        self.get_logger().info("")
        self.get_logger().info("   Para testar:")
        self.get_logger().info("   - Certifique-se que o fase1.py está rodando")
        self.get_logger().info("   - O fase1.py deve mostrar 'Detectadas X bases!' e iniciar visitação")
        
        # Continua publicando a cada 10 segundos para garantir que seja recebido
        self.create_timer(10.0, self.republish_bases)

    def republish_bases(self):
        """Republica as bases periodicamente para garantir recepção"""
        
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        3
        for x, y, z in self.base_positions:
            pose = Pose()#!/bin/bash
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        
        self.unique_positions_pub.publish(pose_array)
        self.get_logger().info(f" Republicadas {len(self.base_positions)} bases (para garantir recepção)")

def main(args=None):
    print("   FAKE BASE PUBLISHER")
    print("=" * 50)
    print("   Este script publica as 5 bases diretamente no /unique_positions")
    print("   Usado para testar o algoritmo de pouso do fase1.py")
    print("   Como usar:")
    print("   1. Execute o fase1.py: ros2 run uav_mission fase1")
    print("   2. Execute este script: ros2 run base_detection publish_fake_bases")
    print("   3. O fase1.py deve receber as bases e iniciar visitação")
    print("=" * 50)
    print()
    
    rclpy.init(args=args)
    publisher = FakeBasePublisher()
    
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info(" Fake Base Publisher interrompido pelo usuário")
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 