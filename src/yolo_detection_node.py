#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose

class YOLODetectionNode:
    def __init__(self):
        rospy.init_node('yolo_detection_node', anonymous=True)
        
        # Parâmetros ROS
        self.model_path = rospy.get_param('~model_path', 'best.pt')
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.25)
        self.camera_id = rospy.get_param('~camera_id', 0)
        
        # Publishers
        self.detection_pub = rospy.Publisher('/vision/detections', String, queue_size=10)
        self.pose_array_pub = rospy.Publisher('/vision/detection_poses', PoseArray, queue_size=10)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Bridge para conversão de imagens
        self.bridge = CvBridge()
        
        # Carregar modelo YOLO
        try:
            # Tentar caminhos relativos e absolutos
            model_paths = [
                self.model_path,
                os.path.join(os.path.dirname(__file__), self.model_path),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "Modelo Visão Computacional", self.model_path)
            ]
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    self.model = YOLO(path)
                    rospy.loginfo(f"Modelo YOLO carregado: {path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                rospy.logerr(f"Modelo não encontrado. Tentou: {model_paths}")
                return
                
        except Exception as e:
            rospy.logerr(f"Erro ao carregar modelo: {e}")
            return
        
        # Não inicializar câmera diretamente - usar ROS tópicos
        rospy.loginfo("YOLO Detection Node iniciado - aguardando imagens via ROS")
        
    def image_callback(self, msg):
        """Callback para processar imagens recebidas via ROS"""
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frame(cv_image)
            
            # Mostrar imagem (opcional)
            if rospy.get_param('~show_image', False):
                cv2.imshow('YOLO Detection', cv_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User requested shutdown")
                    
        except Exception as e:
            rospy.logerr(f"Erro no callback de imagem: {e}")
    
    def process_frame(self, frame):
        """Processa um frame e publica detecções"""
        # Realizar inferência
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        r = results[0]
        
        # Processar detecções
        detections = []
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "camera_frame"
        
        if r.boxes:
            for box in r.boxes:
                class_id = int(box.cls.item())
                class_name = r.names[class_id]
                conf = box.conf.item()
                bbox = box.xyxy.tolist()
                
                # Criar mensagem de detecção
                detection_msg = f"Class: {class_name}, Confidence: {conf:.2f}, BBox: {bbox}"
                detections.append(detection_msg)
                
                # Criar pose para cada detecção
                pose = Pose()
                pose.position.x = (bbox[0] + bbox[2]) / 2  # Centro X
                pose.position.y = (bbox[1] + bbox[3]) / 2  # Centro Y
                pose.position.z = conf  # Usar confiança como Z
                pose_array.poses.append(pose)
                
                rospy.loginfo(f"Detectado: {detection_msg}")
        
        # Publicar detecções
        if detections:
            self.detection_pub.publish(f"Detections: {len(detections)} objects")
            self.pose_array_pub.publish(pose_array)
        else:
            self.detection_pub.publish("No detections")
    
    def run(self):
        """Executa o nó - processamento via callbacks ROS"""
        rospy.loginfo("YOLO Detection Node rodando - aguardando imagens...")
        
        while not rospy.is_shutdown():
            # Processamento é feito via callbacks ROS
            rate = rospy.Rate(10)  # 10 Hz para verificação
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = YOLODetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 