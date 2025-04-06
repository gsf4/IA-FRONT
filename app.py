import gradio as gr
import requests

# Função para enviar imagem para a API e obter o resultado
def classify_image(image):
    url = "http://127.0.0.1:7860/predict"  # Substitua pelo endpoint da API
    files = {"file": ("image.jpg", open(image, "rb"), "image/jpeg")}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        result = response.json()
        return "Magic: The Gathering card!" if result["isMagicCard"] else "Not a Magic card."
    else:
        return "Error: Unable to process the image."

# Interface Gradio
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Magic: The Gathering Card Detector",
    description="Upload a JPEG image to check if it's a Magic: The Gathering card."
)

# Exportar como página estática
interface.launch(share=True)
print("deu bom meu caro")
