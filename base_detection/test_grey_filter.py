import cv2
import numpy as np
from ultralytics import YOLO
import torch

# ISSO AQUI E SO UM TESTE PARA VER SE O FILTRO CINZA ESTA RODANDO BEM 
# AQUI A GENTE PEGA INFORMACAO VISUAL DA WEBCAM, E NECESSARIO VALIDAR MELHRO E VALIDAR O SISTEMA DE CORRECAO DE POSICAO
# NÂO TESTEI NA SIMULACAO


def create_color_mask(frame):
    """
    Cria uma máscara combinada para detectar azul e amarelo.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Máscara para azul
    blue_lower = np.array([100, 50, 50], dtype=np.uint8)
    blue_upper = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    
    # Máscara para amarelo
    yellow_lower = np.array([20, 50, 50], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Combina as máscaras
    combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)
    
    return combined_mask

def calculate_adaptive_hsv(frame):
    """
    Calcula parâmetros HSV para detectar base azul e amarela.
    """
    # Converte para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calcula estatísticas dos canais HSV
    h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
    h_std, s_std, v_std = np.std(hsv, axis=(0, 1))
    
    # Para detectar azul e amarelo, usamos faixas específicas
    # Azul: H ~ 120, Amarelo: H ~ 30
    # Vamos usar uma faixa ampla que capture ambas as cores
    h_lower = 20   # Captura amarelo e laranja
    h_upper = 140  # Captura azul e verde-azulado
    
    # Saturação: cores vivas têm alta saturação
    s_lower = max(50, int(s_mean - s_std))
    s_upper = min(255, int(s_mean + s_std))
    
    # Valor: ajusta baseado na iluminação
    v_lower = max(50, int(v_mean - v_std))
    v_upper = min(255, int(v_mean + v_std))
    
    # Converte para uint8 para compatibilidade com OpenCV
    hsv_lower = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
    hsv_upper = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)
    
    return hsv_lower, hsv_upper

def is_rectangular(contour, width, height):
    """
    Verifica se um contorno é minimamente retangular.
    """
    # Calcula a área do contorno
    contour_area = cv2.contourArea(contour)
    
    # Calcula a área do bounding box
    bbox_area = width * height
    
    # Se a área do contorno for muito menor que a do bbox, não é retangular
    if contour_area < 0.3 * bbox_area:
        return False
    
    # Calcula a razão entre largura e altura
    aspect_ratio = max(width, height) / min(width, height)
    
    # Se for muito alongado (como uma linha), não é um pacote
    if aspect_ratio > 5:
        return False
    
    # Se for muito pequeno em uma dimensão, pode ser ruído
    if min(width, height) < 10:
        return False
    
    return True

def detect_gray_package(frame, base_bbox):
    """
    Detecta o pacote cinza dentro da área da base detectada.
    """
    x1, y1, x2, y2 = base_bbox
    
    # Recorta a área da base
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    
    if roi.size == 0:
        return None
    
    # Converte para HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Cria múltiplas máscaras para diferentes tons de cinza
    gray_masks = []
    
    # Máscara 1: Cinza muito escuro a escuro
    gray1_lower = np.array([0, 0, 20], dtype=np.uint8)
    gray1_upper = np.array([180, 60, 120], dtype=np.uint8)
    gray_masks.append(cv2.inRange(hsv_roi, gray1_lower, gray1_upper))
    
    # Máscara 2: Cinza escuro a médio
    gray2_lower = np.array([0, 0, 40], dtype=np.uint8)
    gray2_upper = np.array([180, 50, 160], dtype=np.uint8)
    gray_masks.append(cv2.inRange(hsv_roi, gray2_lower, gray2_upper))
    
    # Máscara 3: Cinza médio a claro
    gray3_lower = np.array([0, 0, 60], dtype=np.uint8)
    gray3_upper = np.array([180, 40, 200], dtype=np.uint8)
    gray_masks.append(cv2.inRange(hsv_roi, gray3_lower, gray3_upper))
    
    # Máscara 4: Cinza claro a muito claro (para objetos reflexivos)
    gray4_lower = np.array([0, 0, 100], dtype=np.uint8)
    gray4_upper = np.array([180, 30, 240], dtype=np.uint8)
    gray_masks.append(cv2.inRange(hsv_roi, gray4_lower, gray4_upper))
    
    # Combina todas as máscaras
    combined_mask = gray_masks[0]
    for mask in gray_masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Aplica operações morfológicas para melhorar a detecção
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Encontra contornos
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontra o maior contorno (provavelmente o pacote)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filtra por área mínima (reduzida para capturar objetos menores)
        if cv2.contourArea(largest_contour) > 50:
            # Calcula bounding box do pacote
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Valida se é minimamente retangular
            if is_rectangular(largest_contour, w, h):
                # Converte coordenadas para o frame original
                package_x1 = int(x1) + x
                package_y1 = int(y1) + y
                package_x2 = int(x1) + x + w
                package_y2 = int(y1) + y + h
                
                return [package_x1, package_y1, package_x2, package_y2]
    
    return None

def main():
    # --- Configuração Inicial ---
    # Caminho absoluto para o modelo YOLO
    MODEL_PATH = 'C:/Users/pedro/Downloads/base_detection-main/base_detection-main/base_detection/best.pt'
    
    # Threshold de confiança alto (igual ao usado no base_detection.py)
    DETECTION_THRESHOLD = 0.9
    
    # Verifica se a GPU está disponível e define o dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    print(f"Threshold de detecção: {DETECTION_THRESHOLD}")

    # Carrega o modelo YOLO no dispositivo especificado
    model = YOLO(MODEL_PATH)
    model.to(device)

    # Inicia a captura de vídeo da primeira webcam encontrada
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    print("\nPressione 'q' na janela da câmera para sair do script.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        # --- Pré-processamento com Máscara de Cores Específica ---
        # Cria máscara combinada para azul e amarelo
        mask = create_color_mask(frame)

        # Cria uma imagem resultante com pixels brancos onde a máscara é válida
        result = np.zeros_like(frame)
        result[mask > 0] = [255, 255, 255]

        # --- Detecção com YOLO no frame pré-processado ---
        # Roda a inferência do YOLO no frame pré-processado
        results = model(result, verbose=False)
        base_detections = results[0].boxes.data.tolist()

        # Contador de detecções válidas
        valid_detections = 0

        # Itera sobre todas as bases detectadas
        for detection in base_detections:
            x1, y1, x2, y2, score, class_id = detection
            
            # Filtra apenas detecções com score alto
            if score > DETECTION_THRESHOLD:
                valid_detections += 1
                
                # Calcula o centro da detecção
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # --- Refinamento do Centróide ---
                try:
                    # Recorta a máscara HSV para a área da detecção
                    roi = mask[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Calcula os momentos da máscara recortada
                    moments = cv2.moments(roi)
                    if moments["m00"] > 0:
                        # Calcula o centróide e converte para coordenadas globais
                        c_x = int(moments["m10"] / moments["m00"]) + int(x1)
                        c_y = int(moments["m01"] / moments["m00"]) + int(y1)
                        center_x, center_y = float(c_x), float(c_y)
                except Exception as e:
                    print(f"Falha no cálculo do centróide: {e}. Usando centro do bbox.")
                
                # Desenha o bounding box da base na imagem original (em verde)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                
                # Desenha o centróide refinado (ponto vermelho)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                
                # Mostra o score de confiança
                cv2.putText(frame, f'Base Score: {score:.3f}', (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # --- Detecção do Pacote Cinza ---
                package_bbox = detect_gray_package(frame, [x1, y1, x2, y2])
                if package_bbox:
                    px1, py1, px2, py2 = package_bbox
                    
                    # Desenha o bounding box do pacote (em azul)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 3)
                    
                    # Calcula e mostra o centro do pacote
                    package_center_x = (px1 + px2) / 2
                    package_center_y = (py1 + py2) / 2
                    cv2.circle(frame, (int(package_center_x), int(package_center_y)), 3, (255, 0, 0), -1)
                    
                    # Mostra label do pacote
                    cv2.putText(frame, 'Pacote', (px1, py1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    # Mostra área do pacote para debug
                    area = (px2 - px1) * (py2 - py1)
                    aspect_ratio = max(px2 - px1, py2 - py1) / min(px2 - px1, py2 - py1)
                    cv2.putText(frame, f'Area: {area}', (px1, py2 + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'Ratio: {aspect_ratio:.1f}', (px1, py2 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Mostra informações na tela
        cv2.putText(frame, f'Detecções válidas: {valid_detections}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Threshold: {DETECTION_THRESHOLD}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Filtro: Azul + Amarelo', (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Pacote: Multi-cinza', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostra o resultado final
        cv2.imshow('YOLO Detection - High Confidence Only', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Script finalizado.")

if __name__ == '__main__':
    main() 