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
        """
        Args:
            root (str): Path to the dataset folder. E.g. 'path/to/monet2photo'
            transforms_ (list or None): List of torchvision transforms. 
                                        If None, no transforms are applied.
            unaligned (bool): If True, picks a random B image instead of matching
                              the index with the A image.
            mode (str): 'train' or 'test'. This determines whether we read from
                        trainA/trainB or testA/testB.
        """
        # If no transform list is provided, create an empty transform Compose
        self.transform = transforms.Compose(transforms_ if transforms_ else [])
        self.unaligned = unaligned
        
        # Build file lists from, e.g., root/trainA or root/testA
        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A", "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B", "*.*")))

    def __getitem__(self, index):
        """
        Returns:
            A dictionary with keys {"A", "B"} containing the transformed images.
        """
        # Load image A
        image_A_path = self.files_A[index % len(self.files_A)]
        image_A = Image.open(image_A_path)

        # Load image B (random index if unaligned is True)
        if self.unaligned:
            random_index = random.randint(0, len(self.files_B) - 1)
            image_B_path = self.files_B[random_index]
        else:
            image_B_path = self.files_B[index % len(self.files_B)]
        image_B = Image.open(image_B_path)

        # Convert grayscale to RGB if needed
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        # Apply any transforms provided
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        # The length is the maximum of the two so we don't run out of either
        return max(len(self.files_A), len(self.files_B))
