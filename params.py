"""Неизменяемые параметры"""
import numpy as np
from pathlib import Path
import torch.cuda

# Пути
DATASET_PATH = Path('images')
TEST_DATASET_PATH = Path('test_images')
MODEL_SAVE_PATH = Path('Models')
PATCH_DIR_NAME = 'patches'

# Параметры данных
PATCH_SIZE = (512, 512)     # Базовый размер изображения, которое принимает нейросеть.
IMAGE_EXTS = ('.png', '.PNG', '.tiff', '.TIFF', '.jpeg', '.jpg', '.JPEG', '.JPG')   #
VAL_SPLIT = 0.15    #    Доля Изображений, которая будет отобрана из основного датасета как валидационные.
BATCH_SIZE = 32     #   от 4 до 32 в зависимости от памяти GPU и модели.
NUM_WORKERS = 4     # Количество работников, используемых при параллельных вычислениях.
N_REPEATS = 32      # Множитель датасета, т.е. 32 означает что датасет в 32 раза больше (синтетические данные)

# Параметры модели
ENCODER_CHANNELS = [16, 32, 64, 128, 256, 512]      # Количество слоёв в UNet энкодере и декодере.
IN_CHANNELS = 1     # Количество каналов цвета. 3 - RGB, 1 - grayscale
OUT_CLASSES = 4     # Количество классов для детекции.

# Цветовая схема
COLORS_CONFIG = [
    {'name': 'red',
     'rgb': np.array([255, 0, 0], dtype='uint8'),
     'class': 0},
    {'name': 'yellow',
     'rgb': np.array([255, 216, 0], dtype='uint8'),
     'class': 1},
    {'name': 'green',
     'rgb': np.array([0, 255, 0], dtype='uint8'),
     'class': 2},
    {'name': 'black',
     'rgb': np.array([0, 0, 0], dtype='uint8'),
     'class': 3},
]


# Настройки обучения
NUM_EPOCHS = 2500   # Количество эпох обучения
LEARNING_RATE = 1e-5    # Начальное значение learning rate. Зависит от модели, в среднем в диапазоне от 1e-3 до 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # использует cuda Nvidia, если доступен. (AMD не проверялось)