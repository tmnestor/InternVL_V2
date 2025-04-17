"""
Logging utilities.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Set up a logger for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if specified
    if log_file is not None:
        if isinstance(log_file, str):
            log_file = Path(log_file)
            
        # Create directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Create a default log file in the archive location
        log_dir = Path("archive/logging/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorboardLogger:
    """
    Wrapper for TensorBoard logging functionality.
    """
    def __init__(self, log_dir: Union[str, Path] = None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs. If None, uses archive/logging/logs/tensorboard
        """
        if log_dir is None:
            log_dir = Path("archive/logging/logs/tensorboard")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            print("TensorBoard not found. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value.
        
        Args:
            tag: Name for the value
            value: The value to log
            step: Step number
        """
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """
        Log multiple scalar values.
        
        Args:
            main_tag: Group name for the values
            tag_scalar_dict: Dictionary of tag names and values
            step: Step number
        """
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_image(self, tag: str, img_tensor, step: int) -> None:
        """
        Log an image.
        
        Args:
            tag: Name for the image
            img_tensor: Image tensor to log (CHW format)
            step: Step number
        """
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_histogram(self, tag: str, values, step: int) -> None:
        """
        Log histogram of values.
        
        Args:
            tag: Name for the histogram
            values: Values to create histogram from
            step: Step number
        """
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self) -> None:
        """Close the TensorBoard logger."""
        if self.enabled:
            self.writer.close()