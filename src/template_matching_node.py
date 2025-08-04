#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray, Pose

class TemplateMatchingNode:
    def __init__(self):
        rospy.init_node('template_matching_node', anonymous=True)
        
        # Parâmetros ROS
        self.template_dir = rospy.get_param('~template_dir', 'Template')
        self.algorithm = rospy.get_param('~algorithm', 'AKAZE')  # SIFT, ORB, AKAZE
        self.camera_id = rospy.get_param('~camera_id', 0)
        self.min_matches = rospy.get_param('~min_matches', 10)
        self.min_inliers = rospy.get_param('~min_inliers', 30)
        
        # Publishers
        self.matching_pub = rospy.Publisher('/vision/template_matches', String, queue_size=10)
        self.pose_array_pub = rospy.Publisher('/vision/template_poses', PoseArray, queue_size=10)
        
        # Subscribers
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        # Bridge para conversão de imagens
        self.bridge = CvBridge()
        
        # Inicializar detector
        self.setup_detector()
        
        # Carregar templates
        self.templates = self.load_templates()
        
        # Não inicializar câmera diretamente - usar ROS tópicos
        rospy.loginfo("Template Matching Node iniciado - aguardando imagens via ROS")
        
    def setup_detector(self):
        """Configura o detector baseado no algoritmo escolhido"""
        if self.algorithm == 'SIFT':
            self.detector = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif self.algorithm == 'AKAZE':
            self.detector = cv2.AKAZE_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:  # ORB
            self.detector = cv2.ORB_create(nfeatures=1500)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        rospy.loginfo(f"Detector configurado: {self.algorithm}")
    
    def load_templates(self):
        """Carrega templates da pasta especificada"""
        templates = []
        
        if not os.path.exists(self.template_dir):
            rospy.logwarn(f"Pasta de templates não encontrada: {self.template_dir}")
            return templates
        
        for file_name in os.listdir(self.template_dir):
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(self.template_dir, file_name)
                img = cv2.imread(path, 0)
                if img is not None:
                    kp, des = self.detector.detectAndCompute(img, None)
                    templates.append({
                        'name': os.path.splitext(file_name)[0],
                        'image': img,
                        'keypoints': kp,
                        'descriptors': des
                    })
                    rospy.loginfo(f"Template carregado: {file_name} ({len(kp)} keypoints)")
                else:
                    rospy.logwarn(f"Falha ao carregar {file_name}")
        
        rospy.loginfo(f"Total de templates carregados: {len(templates)}")
        return templates
    
    def image_callback(self, msg):
        """Callback para processar imagens recebidas via ROS"""
        try:
            # Converter ROS Image para OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frame(cv_image)
            
            # Mostrar imagem (opcional)
            if rospy.get_param('~show_image', False):
                cv2.imshow(f'Template Matching - {self.algorithm}', cv_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.signal_shutdown("User requested shutdown")
                    
        except Exception as e:
            rospy.logerr(f"Erro no callback de imagem: {e}")
    
    def process_frame(self, frame):
        """Processa um frame e busca matches com templates"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.detector.detectAndCompute(gray, None)
        
        matches_found = []
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "camera_frame"
        
        for tpl in self.templates:
            if tpl['descriptors'] is None or des_frame is None:
                continue
            
            # Encontrar matches
            matches = self.bf.match(tpl['descriptors'], des_frame)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Pegar melhores matches
            good_matches = matches[:50]
            
            if len(good_matches) > self.min_matches:
                src_pts = np.float32([tpl['keypoints'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Encontrar homografia
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    matches_mask = mask.ravel().tolist()
                    inliers = sum(matches_mask)
                    
                    if inliers > self.min_inliers:
                        # Calcular bounding box
                        h, w = tpl['image'].shape
                        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        
                        # Calcular centro
                        center_x = np.mean(dst[:, 0, 0])
                        center_y = np.mean(dst[:, 0, 1])
                        
                        # Criar pose
                        pose = Pose()
                        pose.position.x = center_x
                        pose.position.y = center_y
                        pose.position.z = inliers / len(good_matches)  # Qualidade do match
                        pose_array.poses.append(pose)
                        
                        matches_found.append(tpl['name'])
                        
                        # Desenhar bounding box
                        frame = cv2.polylines(frame, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3)
                        cv2.putText(frame, tpl['name'], tuple(np.int32(dst[0][0])), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        rospy.loginfo(f"Template detectado: {tpl['name']} (inliers: {inliers})")
        
        # Publicar resultados
        if matches_found:
            self.matching_pub.publish(f"Matches: {', '.join(matches_found)}")
            self.pose_array_pub.publish(pose_array)
        else:
            self.matching_pub.publish("No matches")
    
    def run(self):
        """Executa o nó - processamento via callbacks ROS"""
        rospy.loginfo("Template Matching Node rodando - aguardando imagens...")
        
        while not rospy.is_shutdown():
            # Processamento é feito via callbacks ROS
            rate = rospy.Rate(10)  # 10 Hz para verificação
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        node = TemplateMatchingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 