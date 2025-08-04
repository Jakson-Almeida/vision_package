#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import glob
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

class CameraCalibrationNode:
    def __init__(self):
        rospy.init_node('camera_calibration_node', anonymous=True)
        
        # Parâmetros ROS
        self.checkerboard = rospy.get_param('~checkerboard', [7, 7])
        self.square_size = rospy.get_param('~square_size_mm', 21.25)
        self.images_path = rospy.get_param('~images_path', 'imagens_calibracao')
        
        # Publishers
        self.calibration_status_pub = rospy.Publisher('/vision/calibration_status', String, queue_size=1)
        
        # Bridge para conversão de imagens
        self.bridge = CvBridge()
        
        # Critérios para sub-pixels
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        rospy.loginfo("Camera Calibration Node iniciado")
        
    def calibrate_camera(self):
        """Realiza a calibração da câmera"""
        rospy.loginfo("Iniciando calibração da câmera...")
        
        # Preparar pontos 3D do objeto
        objp = np.zeros((1, self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        objp = objp * self.square_size
        
        # Vetores para armazenar pontos
        objpoints = []
        imgpoints = []
        
        # Buscar imagens de calibração
        images = glob.glob(f"{self.images_path}/*.jpg")
        
        if not images:
            rospy.logerr(f"Nenhuma imagem encontrada em {self.images_path}")
            self.calibration_status_pub.publish("ERROR: No images found")
            return False
            
        rospy.loginfo(f"Encontradas {len(images)} imagens para calibração")
        
        # Processar cada imagem
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                rospy.logwarn(f"Não foi possível carregar {fname}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Encontrar cantos do tabuleiro
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                
                # Desenhar cantos
                img = cv2.drawChessboardCorners(img, self.checkerboard, corners2, ret)
                cv2.imshow('Calibracao - Cantos Encontrados', img)
                cv2.waitKey(500)
                
                rospy.loginfo(f"Cantos encontrados em {fname}")
            else:
                rospy.logwarn(f"Cantos NÃO encontrados em {fname}")
        
        cv2.destroyAllWindows()
        
        if not objpoints:
            rospy.logerr("Nenhum conjunto de cantos foi encontrado")
            self.calibration_status_pub.publish("ERROR: No corners found")
            return False
        
        # Calibrar câmera
        rospy.loginfo("Realizando calibração...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        # Salvar parâmetros
        np.save("camera_matrix.npy", camera_matrix)
        np.save("dist_coeffs.npy", dist_coeffs)
        
        rospy.loginfo(f"Erro de Re-projeção: {ret}")
        rospy.loginfo(f"Matriz Intrínseca:\n{camera_matrix}")
        rospy.loginfo(f"Coeficientes de Distorção:\n{dist_coeffs}")
        
        self.calibration_status_pub.publish("SUCCESS: Calibration completed")
        return True
    
    def run(self):
        """Executa o nó"""
        rate = rospy.Rate(1)  # 1 Hz
        
        while not rospy.is_shutdown():
            # Verificar se deve realizar calibração
            if rospy.get_param('~perform_calibration', False):
                self.calibrate_camera()
                rospy.set_param('~perform_calibration', False)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        node = CameraCalibrationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 