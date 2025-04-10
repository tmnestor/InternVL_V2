"""
Receipt image processing utilities.

Handles transformations and augmentations for receipt images.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image


class ReceiptProcessor:
    """
    Processor for receipt images with support for augmentations.
    
    Prepares images for the InternVL2 model with appropriate sizing
    and normalization.
    """
    def __init__(self, augment=False, image_size=448):
        """
        Initialize the receipt processor.
        
        Args:
            augment: Whether to apply data augmentation
            image_size: Target image size (default: 448 for InternVL2)
        """
        # Set image size directly for InternVL2
        self.image_size = image_size
        
        # Create transformation pipeline
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if augment:
            self.transform = transforms.Compose([
                # Data augmentation for training
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomAffine(
                    degrees=30,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize,
            ])
    
    def process_image(self, image):
        """
        Process a single image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Processed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image or path to an image")
        
        return self.transform(image)
    
    def process_batch(self, images):
        """
        Process a batch of images.
        
        Args:
            images: List of PIL Images or paths to images
            
        Returns:
            Batch tensor of processed images
        """
        processed = []
        for img in images:
            processed.append(self.process_image(img))
        
        return torch.stack(processed)


def create_blank_image(width=448, height=448, color='white'):
    """
    Create a blank image with the specified dimensions and color.
    
    Args:
        width: Image width
        height: Image height
        color: Background color
        
    Returns:
        PIL Image
    """
    return Image.new('RGB', (width, height), color)