from collections.abc import Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import params
from typing import List, Dict
import os

class DatasetLoader:
    """Класс для загрузки и предобработки dataset"""
    def __init__(self, dataset_path: Path, debug: bool= False):
        self.dataset_path = dataset_path
        self.patch_dir = dataset_path / params.PATCH_DIR_NAME
        self.original_paths = self._get_sorted_paths(dataset_path / 'original')
        self.mask_paths = self._get_sorted_paths(dataset_path / 'mask')
        self.debug = debug
        self.exts = params.IMAGE_EXTS

    def _get_sorted_paths(self, directory: Path) -> List[Path]:
        """Возвращает отсортированный список путей к изображениям"""
        return sorted(
            [p for p in directory.iterdir() if p.suffix.lower() in params.IMAGE_EXTS],
            key=lambda x: x.name
        )

    def prepare_dataset(self) -> Dict[str, List[Path]]:
        """Подготавливает dataset с патчами"""
        if not self._check_existing_folders():
            self._process_images()
        if not self._check_existing_patches():
            self._process_images()
        return self._get_patch_paths()


    def get_base_images(self, suffix: str):
        dir = (self.patch_dir / suffix)
        base_names = set()
        for filename in os.listdir(dir):
            if filename.endswith(self.exts):
                base_name = filename.split('_x')[0]
                base_names.add(f'{base_name}{os.path.splitext(filename)[-1]}')

        return base_names

    def _check_existing_folders(self) -> bool:
        """Проверяет существование предобработанных патчей"""
        return (self.patch_dir / 'original').exists() and (self.patch_dir / 'mask').exists()

    def _check_existing_patches(self):
        existing_patches = dict()

        # находим существующие патчи и все существующие изображения
        existing_patches['original'] = self.get_base_images('original')
        existing_patches['mask'] = self.get_base_images('mask')
        all_images = {os.path.basename(p) for p in self.original_paths}
        all_masks = {os.path.basename(p) for p in self.mask_paths}

        # Находим изображения, для которых нет патчей
        missing_images = all_images - existing_patches['original']
        missing_masks = all_masks - existing_patches['mask']

        if missing_images or missing_masks:
            if self.debug:
                print(
                    f"Missing patches for {len(missing_images)} images and {len(missing_masks)} masks has been detected.")
            return False


        return True

    def _process_images(self):
        """Обрабатывает изображения и создаёт патчи"""
        if self.debug:
            print(f"Starting to process images")
        with ThreadPoolExecutor() as executor:
            executor.map(self._process_single_image, ['original', 'mask'])

    def _process_single_image(self, suffix: str):
        """Обрабатывает изображения одного типа (original/mask)"""
        output_dir = self.patch_dir / suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in getattr(self, f'{suffix}_paths'):
            with Image.open(path) as img:
                self._create_patches(img, path, output_dir)

    def _create_patches(self, img: Image.Image, src_path: Path, output_dir: Path):
        width, height = img.size
        base_name = src_path.stem

        for y in range(0, height, params.PATCH_SIZE[1]):
            for x in range(0, height, params.PATCH_SIZE[0]):
                patch = img.crop((x, y, x + params.PATCH_SIZE[0], y + params.PATCH_SIZE[1]))
                patch_name = f"{base_name}_x{x}_y{y}{src_path.suffix}"
                patch.save(output_dir / patch_name)

    def _get_patch_paths(self) -> Dict[str, List[Path]]:
        """Возвращает пути к подготовленным патчам"""
        return {
            'original': sorted((self.patch_dir / 'original').glob('*')),
            'mask': sorted((self.patch_dir / 'mask').glob('*'))
        }


class SEMDataset(Dataset):
    def __init__(self,
                 original_paths: List[Path],
                 mask_paths: List[Path],
                 transform: Callable=None,
                 grayscale: bool=False,
                 n_repeats=params.N_REPEATS
                 ):
        self.original_paths = original_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.grayscale = grayscale
        self.n_repeats = n_repeats
        # self.color_to_class = {tuple(color['rgb']): color['class'] for color in params.COLORS_CONFIG}

    def __len__(self) -> int:
        return len(self.original_paths) * self.n_repeats

    def __getitem__(self, idx):
        original_idx = idx // self.n_repeats
        image_path = self.original_paths[original_idx]
        mask_path = self.mask_paths[original_idx]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].numpy()
        return image, self._convert_mask_to_labels(mask)

    def _load_image(self, path: Path) -> np.ndarray:
        """Загружает изображения в нужном формате"""
        mode = "L" if self.grayscale else "RGB"
        return np.array(Image.open(path).convert(mode))

    def _load_mask(self, path: Path) -> np.ndarray:
        """Загружает маску изображения в нужном формате"""
        return np.array(Image.open(path).convert("RGB"))

    def _convert_mask_to_labels(self, mask: np.ndarray) -> torch.Tensor:
        """Конвертирует RGB маску в тензор меток классов"""
        ref_colors = np.array([color['rgb'] for color in params.COLORS_CONFIG], dtype=np.int32)
        distances = np.linalg.norm(mask[:, :, None] - ref_colors, axis=3)
        closest_color_indices = np.argmin(distances, axis=2)

        label_mask = np.full(mask.shape[:2], 3, dtype=np.uint8)
        for i, color_info in enumerate(params.COLORS_CONFIG):
            label_mask[closest_color_indices == i] = color_info['class']

        return torch.from_numpy(label_mask).long()
