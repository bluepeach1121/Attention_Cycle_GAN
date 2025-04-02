import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_ if transforms_ else [])
        self.unaligned = unaligned
        
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A", "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B", "*.*")))

    def __getitem__(self, index):
        image_A_path = self.files_A[index % len(self.files_A)]
        image_A = Image.open(image_A_path)

        if self.unaligned:
            random_index = random.randint(0, len(self.files_B) - 1)
            image_B_path = self.files_B[random_index]
        else:
            image_B_path = self.files_B[index % len(self.files_B)]
        image_B = Image.open(image_B_path)

        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
