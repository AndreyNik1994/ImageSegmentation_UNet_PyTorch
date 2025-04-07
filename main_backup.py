import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import MetricCollection, JaccardIndex, Accuracy, Precision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import LinearLR

import params
from dataset_loader import SEMDataset, DatasetLoader
from model import UNet, TransUNet
from general_func import save_model, convert_labels_to_rgb
from transforms import get_train_transforms, get_test_transforms
from params import *

writer = SummaryWriter(log_dir='runs/transunet_experiment')

# Инициализация метрик
def init_metrics() -> MetricCollection:
    return MetricCollection({
        'iou': JaccardIndex(task='multiclass', num_classes=OUT_CLASSES, ignore_index=3),
        'accuracy': Accuracy(task='multiclass', num_classes=OUT_CLASSES, ignore_index=3),
    }).to(DEVICE)

def train_model():
    # Инициализация хранилища метрик
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_acc': [],
        'val_acc': [],
    }

    # Инициализация модели
    model = UNet(
        in_channels=IN_CHANNELS,
        out_classes=OUT_CLASSES,
        encoder_channels=ENCODER_CHANNELS
    ).to(DEVICE)

    """Если хотите продолжить обучение модели, то необходимо активировать строки и внести соответствующий путь до 
    сохранённой модели"""
    # path_to_model_load = Path('Models') / '06_april_2025_Gray_TransUNet' / 'best_epoch917_iou0.73_GRAY_TransUNet_detection.pth'
    # model = general_func.load_model(model, path_to_model_load)

    # Подготовка данных
    loader = DatasetLoader(params.DATASET_PATH, debug=True)
    data = loader.prepare_dataset()
    test_loader = DatasetLoader(params.TEST_DATASET_PATH)
    test_data = test_loader.prepare_dataset()

    # Создание датасетов
    full_dataset = SEMDataset(
        original_paths=data['original'],
        mask_paths=data['mask'],
        transform=get_train_transforms(),
        grayscale=True
    )

    # Разделение на тренировочную и валидационную выборки
    train_size = int(len(full_dataset) * (1 - VAL_SPLIT))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Инициализация компонентов обучения
    criterion = nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    # scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=10)
    scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    train_metrics = init_metrics()
    val_metrics = init_metrics()
    best_iou = 0.0
    loss = None

    # Цикл обучения
    for epoch in range(NUM_EPOCHS):
        print(f'\n------ Epoch {epoch + 1}/{NUM_EPOCHS} ------')

        # Тренировочная фаза
        model.train()
        train_metrics.reset()
        i = 0
        for images, masks in train_loader:
            sys.stdout.write(f'\rBatch {i + 1}/{len(train_loader)}')
            images = images.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE)

            # Forward pass

            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:  # Раз в 10 батчей
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        writer.add_scalar(f'Gradients/Norm/{name}', grad_norm, epoch * len(train_loader) + i)
                        writer.add_histogram(f'Gradients/Values/{name}', param.grad, epoch * len(train_loader) + i)

                # Закрыть writer после обучения

            # Обновление метрик
            preds = torch.argmax(outputs, dim=1)
            train_metrics.update(preds, masks)
            i += 1

        # Вывод метрик обучения
        train_results = train_metrics.compute()
        print(f"\nTrain Loss: {loss.item():.4f} | IoU: {train_results['iou']:.4f} "
              f"| Acc: {train_results['accuracy']:.4f}")

        # Сохранение метрик обучения
        history['train_loss'].append(loss.item())
        history['train_iou'].append(train_results['iou'].item())
        history['train_acc'].append(train_results['accuracy'].item())

        # Валидационная фаза
        model.eval()
        val_metrics.reset()
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, dtype=torch.float32)
                masks = masks.to(DEVICE)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_metrics.update(preds, masks)

        # Вывод метрик валидации
        val_results = val_metrics.compute()
        print(f"Val Loss: {loss.item():.4f} | IoU: {val_results['iou']:.4f} "
              f"| Acc: {val_results['accuracy']:.4f}")

        history['val_loss'].append(loss.item())
        history['val_iou'].append(val_results['iou'].item())
        history['val_acc'].append(val_results['accuracy'].item())


        # Обновление learning rate и сохранение модели
        scheduler.step(val_results['iou'])
        if val_results['iou'] > best_iou:
            best_iou = val_results['iou']
            save_model(model, f'best_epoch{epoch + 1}_iou{best_iou:.2f}_{'RGB' if IN_CHANNELS == 3 else "GRAY"}_TransUNet')

    writer.close()

    # Тестирование и визуализация
    test_dataset = SEMDataset(
        original_paths=test_data['original'],
        mask_paths=test_data['mask'],
        transform=get_test_transforms(),
        grayscale=True,
        n_repeats=1
    )

    plot_training_history(history, params.NUM_EPOCHS)
    visualize_predictions(model, test_dataset)


def plot_training_history(history: dict, current_epoch: int):
    plt.figure(figsize=(15, 10))

    # График потерь
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # График IoU
    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.title('IoU History')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    # График Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.tight_layout()
    plt.savefig(f'training_history_epoch_{current_epoch}.png')
    plt.close()

def visualize_predictions(model: nn.Module, dataset: SEMDataset):
    model.eval()
    plt.figure(figsize=(15, 9))

    for idx in range(4):
        image, true_mask = dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE))
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        # Визуализация
        plt.subplot(4, 3,idx * 3 + 1)
        plt.title("Original")
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())

        plt.subplot(4, 3, idx * 3 +  2)
        plt.title("True Mask")
        plt.imshow(convert_labels_to_rgb(true_mask.numpy()))

        plt.subplot(4, 3, idx * 3 +  3)
        plt.title("Prediction")
        plt.imshow(convert_labels_to_rgb(pred_mask))

        plt.tight_layout()
        plt.savefig('predictions.png')
    plt.show()


if __name__ == '__main__':
    # Создание необходимых директорий
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    train_model()