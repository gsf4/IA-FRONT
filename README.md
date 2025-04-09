# Detector de Cartas com YOLOv11-OBB

Este projeto utiliza **YOLOv11 com caixas delimitadoras orientadas (OBB)** para detectar cartas em imagens e identificar sua **cor/multicor** com base no atributo `colorIdentity`. Ele possui um backend em **FastAPI** e uma interface frontend em **Gradio** para facilitar a visualização e teste do modelo.

---

## 🧩 Dependências

Certifique-se de ter o **Python 3.10+** instalado. Para instalar as bibliotecas necessárias, execute:

```bash
pip install -r requirements.txt
```
Ou, instale manualmente os principais pacotes:

```bash
pip install ultralytics opencv-python fastapi uvicorn gradio pillow
```
Caso utilize CUDA, certifique-se de que o torch esteja com suporte para GPU:

bash

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
bash

Copiar
pip install ultralytics opencv-python fastapi uvicorn gradio pillow
⚠️ Caso utilize CUDA, certifique-se de que o torch esteja com suporte para GPU:

# Instruções - Notebook mtg_detect.ipynb
Este notebook contém os passos para:

- Processamento do dataset (annotation e metadata)
- Conversão para o formato OBB do YOLO
- Treinamento e avaliação do modelo
- Visualização dos resultados

## Para executá-lo:
1. Abra o arquivo mtg_detect.ipynb em Jupyter Notebook ou VS Code.
2. Execute as células sequencialmente.
3. O modelo final será salvo em runs/obb/<nome_da_execução>/weights/best.pt.

# Como Rodar o Backend
Caminho: backend/app.py
1. Atualize o caminho do modelo no arquivo:
```bash
model = YOLO("../runs/obb/yolo_cards_obbX/weights/best.pt")
```

2. Rode o servidor FastAPI:
```bash
uvicorn backend.app:app --reload
```

3. Acesse os endpoints:
- POST /predict: retorna as predições em JSON.
- POST /predict-image: retorna a imagem com as detecções desenhadas.

Por padrão, o serviço roda em http://localhost:8000

# Como Rodar o Frontend
Caminho: frontend/app_gradio.py
Este script consome a API do backend e exibe as imagens com as detecções.

Passos:
Certifique-se de que o backend está rodando em http://localhost:8000.

Execute o script:

```bash
python frontend/app_gradio.py
```

Um link será aberto automaticamente no navegador com a interface.

# Classes Detectadas
O modelo detecta cartas com as seguintes classes de cor (extraídas de colorIdentity):

| Class | Descrição             |
|-------|-----------------------|
| 0     | Branco (White)        |
| 1     | Azul (Blue)           |
| 2     | Preto (Black)         |
| 3     | Vermelho (Red)        |
| 4     | Verde (Green)         |
| 5     | Incolor (Colorless)   |
| 6     | Multicor (Multicolor) |
