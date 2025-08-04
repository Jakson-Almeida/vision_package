# Webcam Calibração - Parâmetros

## 📷 Especificações da Webcam

- **ID**: 0 (webcam principal)
- **Resolução**: 640x480
- **Formato**: JPG
- **Fabricante**: Não especificado

## 🎯 Parâmetros de Calibração

### Matriz Intrínseca (K)
```
[[668.52659926   0.         308.58046185]
 [  0.         667.94509825 225.82804249]
 [  0.           0.           1.        ]]
```

### Coeficientes de Distorção
```
[0.04126039, 0.11358728, -0.00148707, -0.00142178, -0.83070678]
```

## 📊 Métricas de Qualidade

- **Erro de Re-projeção**: 0.216
- **Imagens Utilizadas**: 5
- **Tabuleiro**: 8x8 (7x7 cantos internos)
- **Tamanho do Quadrado**: 21.25mm (2.125cm)

## 💾 Arquivos de Calibração

- `camera_matrix.npy`: Matriz intrínseca
- `dist_coeffs.npy`: Coeficientes de distorção

## 🔧 Uso no Código

```python
import numpy as np

# Carregar parâmetros
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# Usar em funções OpenCV
ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
```

## ✅ Status

- **Calibração**: ✅ Concluída
- **Qualidade**: ✅ Excelente (erro < 1.0)
- **Pronta para uso**: ✅ Sim 