# HandConnect - Protótipo

# Etapas principais:
# 1. Gravação de sinais customizados (etapa atual)
# 2. Reconhecimento de gestos com MediaPipe
# 3. Mapeamento gesto -> texto bruto (pacote de sinais)
# 4. Tradução com API do ChatGPT
# 5. Conversão texto -> fala (TTS)
# 6. Interface gráfica bonita/simples/moderna

import cv2
import mediapipe as mp
import pickle
import os
import shutil

CAMINHO_PACOTE = "pacote_personalizado.pkl"

# Inicialização do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Carregar ou criar pacote
if os.path.exists(CAMINHO_PACOTE):
    with open(CAMINHO_PACOTE, 'rb') as f:
        pacote_sinais = pickle.load(f)
else:
    pacote_sinais = {}


def capturar_landmarks(frame, resultados):
    todas_maos = []
    if resultados.multi_hand_landmarks:
        for idx, mao in enumerate(resultados.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,
                mao,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            landmarks = [(lm.x, lm.y, lm.z) for lm in mao.landmark]
            todas_maos.append(landmarks)
    return todas_maos


def gravar_novo_sinal():
    nome = input("Digite o nome do novo sinal: ")
    if nome in pacote_sinais:
        confirmar = input(f"Sinal '{nome}' já existe. Deseja sobrescrever? (s/n): ")
        if confirmar.lower() != 's':
            print("Operação cancelada.")
            return

    capturas = []
    cap = cv2.VideoCapture(0)
    print("Mostre o sinal para a câmera. Pressione 'c' para capturar, 'q' para cancelar.")

    while len(capturas) < 5:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)
        todas_maos = capturar_landmarks(frame, resultados)

        cv2.putText(frame, f"Capturas: {len(capturas)}/5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Gravando sinal", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('c'):
            if todas_maos:
                # Para gravação, pegamos apenas a primeira mão
                capturas.append(todas_maos[0])
                print(f"Captura {len(capturas)} salva.")
            else:
                print("Nenhuma mão detectada. Tente novamente.")
        elif tecla == ord('q'):
            print("Cancelado.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if capturas:
        media = [
            tuple(sum(coord[i] for coord in pontos) / len(pontos) for i in range(3))
            for pontos in zip(*capturas)
        ]

        # Backup
        if os.path.exists(CAMINHO_PACOTE):
            shutil.copy(CAMINHO_PACOTE, CAMINHO_PACOTE + ".bak")

        pacote_sinais[nome] = media
        with open(CAMINHO_PACOTE, 'wb') as f:
            pickle.dump(pacote_sinais, f)
        print(f"Sinal '{nome}' salvo com sucesso.")


if __name__ == "__main__":
    gravar_novo_sinal()

