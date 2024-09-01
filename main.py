import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms, models
import torch.nn.functional as F  # Импортируем функциональный модуль
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
import numpy as np
from io import BytesIO
import asyncio
import uvicorn  # Импортируем uvicorn для запуска FastAPI
from fastapi.middleware.cors import CORSMiddleware


Debug = False

# Путь к папке с моделями
model_dir = "Models"

# Путь к файлу модели
inception_model_path = os.path.join(model_dir, "inception_v3.pth")
vit_model_path = os.path.join(model_dir, "vit_b_16.pth")

# Глобальные переменные для моделей
inception_model = None
vit_model = None
efficientnet= None

# Функция загрузки или сохранения модели
def load_or_download_model(model_name, model_path, download_func):
    if os.path.exists(model_path):
        print(f"Loading {model_name} from disk...")
        model = torch.load(model_path)
    else:
        print(f"Downloading {model_name} from the internet...")
        model = download_func(pretrained=True)
        torch.save(model, model_path)
    return model


# Инициализация моделей
def init_models(device):
    global inception_model, vit_model, efficientnet

    '''# Загрузка или скачивание InceptionV3
    inception_model = load_or_download_model(
        "InceptionV3", inception_model_path, models.inception_v3
    )
    inception_model.eval()  # Перевод модели в режим оценки

    # Загрузка или скачивание ViT
    vit_model = load_or_download_model(
        "ViT_B_16", vit_model_path, models.vit_b_16
    )
    vit_model.eval()  # Перевод модели в режим оценки'''

    # Используем EfficientNet B7 для извлечения эмбеддингов
    efficientnet = models.efficientnet_b7(pretrained=True).to(device)
    efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])  # Убираем последний слой (классификатор)
    efficientnet.eval()  # Устанавливаем модель в режим оценки

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализируем модели при запуске приложения
init_models(device)

# Загрузка обученной модели
class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)

    def forward(self, x):
        batch_size, _ = x.size()
        # Глобальное усреднение (Global Average Pooling) по оси каналов
        out = self.fc1(x)
        out = F.relu(out)
        out = torch.sigmoid(self.fc2(out))
        return x * out


# Определение простой классификационной модели
class SimpleClassifier(nn.Module):

        def __init__(self, input_dim):
            super(SimpleClassifier, self).__init__()
            self.se_block = SEBlock(input_dim)
            self.fc1 = nn.Linear(input_dim, input_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(input_dim, 1)
            self.fc3 = nn.Linear(128, 1)  # Выходной слой для бинарной классификации (1 логит)
            self.dropout = nn.Dropout(0.2)
            self.bn0 = nn.BatchNorm1d(256)
            self.bn1 = nn.BatchNorm1d(1000)

        def forward(self, x):
            #x = self.se_block(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            '''x = self.relu(x)
            x = self.bn1(x)
            x = self.fc3(x)'''  # Один логит как выход
            return x



model = SimpleClassifier(input_dim=2560)
model_path = os.path.join(model_dir, "clsfr.pth")  # путь к лучшей модели
#model = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
model = model.to(device)
model.eval()

# Преобразования для входного изображения
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# FastAPI приложение
app = FastAPI()

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить доступ со всех источников. Можно ограничить доступ конкретными доменами.
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все HTTP методы.
    allow_headers=["*"],  # Разрешить все заголовки.
)


def extract_embeddings(image, device):
    # Применяем преобразования к изображению

    image = preprocess(image).to(device)
    image = image.unsqueeze(0)  # добавляем batch dimension

    with torch.no_grad():

        #embeddings = features.view(features.size(0), -1).cpu().numpy()
        embeddings = efficientnet(image)
        embeddings = torch.flatten(embeddings, 1)
        #inputs = image.to(device)
        features = model(embeddings)
    return embeddings


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Читаем изображение
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    # Извлекаем эмбеддинги
    embeddings = extract_embeddings(image, device)

    # Классификация
    with torch.no_grad():
        outputs = model(embeddings)
        # Преобразуем логиты в вероятности
        probabilities = torch.sigmoid(outputs)

        # Преобразуем вероятности в предсказания (0 или 1)
        predicted = (probabilities >= 0.5)[0]
        class_idx = predicted.item()

    # Классы: 0 - benign (доброкачественная), 1 - malignant (злокачественная)
    class_names = ["Доброкачественный", "Злокачественный"]
    result = class_names[class_idx]

    return {"class": result}

# Запуск приложения
# Командой в консоли: uvicorn main:app --reload

import io
# Создаем тестовый файл с использованием io.BytesIO
# Функция для отладки с использованием файла на диске
def test_predict_with_file(filepath):
    # Открываем файл на диске в бинарном режиме
    with open(filepath, "rb") as f:
        file_content = f.read()  # Читаем содержимое файла

    # Эмулируем загрузку файла в FastAPI
    file = UploadFile(
        filename=filepath.split("/")[-1],  # Имя файла
        file=io.BytesIO(file_content)  # Содержимое файла
    )

    # Вызываем асинхронную функцию predict
    result = asyncio.run(predict(file))
    print(result)



if __name__ == "__main__":
    if not Debug:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        import io
        # Создаем тестовый файл с использованием io.BytesIO
        # Функция для отладки с использованием файла на диске
        def test_predict_with_file(filepath):
            # Открываем файл на диске в бинарном режиме
            with open(filepath, "rb") as f:
                file_content = f.read()  # Читаем содержимое файла

            # Эмулируем загрузку файла в FastAPI
            file = UploadFile(
                filename=filepath.split("/")[-1],  # Имя файла
                file=io.BytesIO(file_content)  # Содержимое файла
            )

            # Вызываем асинхронную функцию predict
            result = asyncio.run(predict(file))
            print(result)


        # Вызов функции отладки с указанием пути к файлу
        test_predict_with_file("test.jpg")

