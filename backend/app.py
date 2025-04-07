from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
from fastapi.responses import StreamingResponse
import io

model_path = "../runs/obb/yolo_cards_obb9/weights/best.pt"  # Substitua pelo caminho do seu modelo
model_conf = 0.7  # Substitua pela confiança desejada

COLOR_MAP = {
    0: ("White", (255, 255, 255)),
    1: ("Blue", (255, 0, 0)),
    2: ("Black", (0, 0, 0)),
    3: ("Red", (0, 0, 255)),
    4: ("Green", (0, 255, 0)),
    5: ("Colorless", (128, 128, 128)),
    6: ("Multicolor", (255, 255, 0)),
}
app = FastAPI()

# Carrega o modelo
model = YOLO(model_path)  # Substitua pelo caminho do seu modelo

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Salva imagem temporariamente
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Faz predição
        results = model(image)

        detections = []
        for box, prob in zip(results[0].obb.xyxy, results[0].obb.conf):
            x1, y1, x2, y2 = map(float, box[:4])
            conf = float(prob)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "class": "card"
            })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        results = model(image, conf=model_conf)
        obbs = results[0].obb
        annotated_img = image.copy()

        for obb in obbs:
            conf = float(obb.conf[0])
            class_id = int(obb.cls[0])
            pts = obb.xyxyxyxy[0].cpu().numpy().reshape((-1, 1, 2)).astype(np.int32)

            class_name, color = COLOR_MAP.get(class_id, ("Unknown", (0, 255, 255)))

            # Desenha o polígono rotacionado com a cor da classe
            cv2.polylines(annotated_img, [pts], isClosed=True, color=color, thickness=3)

            # Pega um ponto da quina para colocar o texto
            x, y = pts[0][0]

            label = f"{class_name} ({conf:.2f})"
            # Define posição do texto
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x, text_y = x, y - 10 if y - 10 > 10 else y + 20

            # Desenha o retângulo de fundo (com padding)
            cv2.rectangle(
                annotated_img,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width + 6, text_y + 4),
                (0, 0, 0),  
                thickness=cv2.FILLED
            )

            # Escreve o texto por cima
            cv2.putText(
                annotated_img,
                label,
                (text_x + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),  # cor da classe
                2,
                cv2.LINE_AA
            )


        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
