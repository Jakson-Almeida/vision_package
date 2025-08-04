#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header

class PoseEstimationNode:
    def __init__(self):
        rospy.init_node('pose_estimation_node', anonymous=True)
        
        # Parâmetros ROS
        self.camera_id = rospy.get_param('~camera_id', 0)
        self.robot_width = rospy.get_param('~robot_width', 0.22)  # metros
        self.robot_length = rospy.get_param('~robot_length', 0.215)  # metros
        self.camera_height = rospy.get_param('~camera_height', 1.5)  # metros
        
        # Publishers
        self.pose_pub = rospy.Publisher('/vision/robot_pose', PoseStamped, queue_size=10)
        self.status_pub = rospy.Publisher('/vision/pose_status', String, queue_size=10)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Bridge para conversão de imagens
        self.bridge = CvBridge()
        
        # Carregar parâmetros de calibração
        self.load_camera_params()
        
        # Definir pontos 3D do robô
        self.setup_robot_points()
        
        # Não inicializar câmera diretamente - usar ROS tópicos
        rospy.loginfo("Pose Estimation Node iniciado - aguardando imagens via ROS")
        
    def load_camera_params(self):
        """Carrega parâmetros de calibração da câmera"""
        try:
            self.camera_matrix = np.load("camera_matrix.npy")
            self.dist_coeffs = np.load("dist_coeffs.npy")
            rospy.loginfo("Parâmetros de calibração carregados")
        except FileNotFoundError:
            # Usar valores padrão se arquivos não existirem
            self.camera_matrix = np.array([
                [668.52659926, 0.,           308.58046185],
                [0.,           667.94509825, 225.82804249],
                [0.,           0.,           1.]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.array([0.04126039, 0.11358728, -0.00148707, -0.00142178, -0.83070678], dtype=np.float32)
            rospy.logwarn("Usando parâmetros de calibração padrão")
    
    def setup_robot_points(self):
        """Define os pontos 3D do robô"""
        self.object_points = np.array([
            [-self.robot_width/2, -self.robot_length/2, 0],   # Canto inferior frontal esquerdo
            [ self.robot_width/2, -self.robot_length/2, 0],   # Canto inferior frontal direito
            [ self.robot_width/2,  self.robot_length/2, 0],   # Canto inferior traseiro direito
            [-self.robot_width/2,  self.robot_length/2, 0]    # Canto inferior traseiro esquerdo
        ], dtype=np.float32)
        
        rospy.loginfo(f"Pontos 3D do robô definidos: {self.robot_width}m x {self.robot_length}m")
    
    def image_callback(self, msg):
        """Callback para processar imagens recebidas via ROS"""
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frame(cv_image)
            
            # Mostrar imagem (opcional)
            if rospy.get_param('~show_image', False):
                cv2.imshow('Pose Estimation', cv_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User requested shutdown")
                    
        except Exception as e:
            rospy.logerr(f"Erro no callback de imagem: {e}")
    
    def process_frame(self, frame):
        """Processa um frame e estima a pose do robô"""
        # Aqui você precisaria detectar os pontos 2D do robô
        # Por simplicidade, vou usar pontos fixos como exemplo
        # Em uma implementação real, você usaria detecção de objetos
        
        # Pontos 2D detectados (exemplo - substitua por detecção real)
        image_points = np.array([
            [51,  365], # Canto inferior frontal esquerdo
            [149, 365], # Canto inferior frontal direito
            [139, 413], # Canto inferior traseiro direito
            [45,  404]  # Canto inferior traseiro esquerdo
        ], dtype=np.float32)
        
        # Validar pontos
        if len(self.object_points) != len(image_points) or len(self.object_points) < 4:
            self.status_pub.publish("ERROR: Invalid points")
            return
        
        # Resolver PnP
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            self.status_pub.publish("ERROR: PnP failed")
            return
        
        # Converter para matriz de rotação
        R_robot_cam, _ = cv2.Rodrigues(rvec)
        
        # Transformar para frame mundial
        cam_world_R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        cam_world_t = np.array([0.0, 0.0, self.camera_height], dtype=np.float32)
        
        # Compor transformações
        T_cam_world = np.eye(4)
        T_cam_world[:3, :3] = cam_world_R
        T_cam_world[:3, 3] = cam_world_t
        
        T_robot_cam = np.eye(4)
        T_robot_cam[:3, :3] = R_robot_cam
        T_robot_cam[:3, 3] = tvec.flatten()
        
        T_robot_world = T_cam_world @ T_robot_cam
        
        # Extrair posição e orientação
        robot_world_pos = T_robot_world[:3, 3]
        robot_world_R = T_robot_world[:3, :3]
        
        # Criar mensagem ROS
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        
        pose_msg.pose.position.x = robot_world_pos[0]
        pose_msg.pose.position.y = robot_world_pos[1]
        pose_msg.pose.position.z = robot_world_pos[2]
        
        # Converter matriz de rotação para quatérnio
        robot_world_rvec, _ = cv2.Rodrigues(robot_world_R)
        
        # Simplificar: usar apenas o vetor de rotação
        pose_msg.pose.orientation.x = robot_world_rvec[0]
        pose_msg.pose.orientation.y = robot_world_rvec[1]
        pose_msg.pose.orientation.z = robot_world_rvec[2]
        pose_msg.pose.orientation.w = 1.0
        
        # Publicar pose
        self.pose_pub.publish(pose_msg)
        self.status_pub.publish("SUCCESS: Pose estimated")
        
        rospy.loginfo(f"Pose do robô: Pos=({robot_world_pos[0]:.3f}, {robot_world_pos[1]:.3f}, {robot_world_pos[2]:.3f})")
    
    def run(self):
        """Executa o nó - processamento via callbacks ROS"""
        rospy.loginfo("Pose Estimation Node rodando - aguardando imagens...")
        
        while not rospy.is_shutdown():
            # Processamento é feito via callbacks ROS
            rate = rospy.Rate(10)  # 10 Hz para verificação
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = PoseEstimationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 