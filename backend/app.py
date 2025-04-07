from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

# Carrega o modelo
model = YOLO("../runs/obb/yolo_cards_obb9/weights/best.pt")  # Substitua pelo caminho do seu modelo

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
        # Lê imagem
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Roda predição
        results = model(image, conf=0.4)

        # Renderiza a imagem com as detecções
        rendered_img = results[0].plot()

        # Codifica em JPEG para envio
        _, img_encoded = cv2.imencode('.jpg', rendered_img)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
