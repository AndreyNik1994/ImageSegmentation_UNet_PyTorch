import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(512, 512),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),  # Гауссовое размытие
        A.GaussNoise(p=0.2),  # Гауссовый шум
        A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.2,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT,
                           fill_mask=3, fill=3,
                           p=1),
        A.GridDistortion(p=0.3),
        # A.RandomCrop(512, 512, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255),
        ToTensorV2()
    ])

def get_test_transforms():
    return A.Compose([
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
        ToTensorV2()
    ])