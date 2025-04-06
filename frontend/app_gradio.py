import gradio as gr
import requests
from PIL import Image
import io

# Função que envia a imagem para a API FastAPI
def detectar_cartas(imagem):
    buffered = io.BytesIO()
    imagem.save(buffered, format="PNG")
    buffered.seek(0)

    files = {'file': buffered.getvalue()}
    response = requests.post("http://localhost:8000/predict-image", files=files)

    if response.status_code == 200:
        img_resultado = Image.open(io.BytesIO(response.content))
        return img_resultado
    else:
        return "Erro ao conectar com o servidor."

# Interface Gradio
demo = gr.Interface(
    fn=detectar_cartas,
    inputs=gr.Image(type="pil", label="Envie uma imagem"),
    outputs=gr.Image(type="pil", label="Resultado da Detecção"),
    title="Detector de Cartas - YOLOv8 OBB",
    description="Faça upload de uma imagem contendo cartas e veja as detecções do modelo.",
    allow_flagging="never"
)

# Iniciar o app
if __name__ == "__main__":
    demo.launch()
