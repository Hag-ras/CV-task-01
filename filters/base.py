"""
Base abstractions for image filters.
Follows Interface Segregation and Dependency Inversion principles.
"""
from abc import ABC, abstractmethod
import numpy as np


class ImageFilter(ABC):
    """Abstract base class for all image filters."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to an image and return the result."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the filter."""
        ...

    @property
    def description(self) -> str:
        """Optional description for UI display."""
        return ""


class KernelFilter(ImageFilter):
    """Base class for kernel-based (convolution) filters."""

    @abstractmethod
    def get_kernel(self, **kwargs) -> np.ndarray:
        """Return the convolution kernel."""
        ...

    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Manual 2D convolution using numpy (no cv2.filter2D).
        Handles both grayscale and multi-channel images.
        Uses reflect padding to avoid border artifacts.
        """
        if image.ndim == 2:
            return self._convolve_channel(image, kernel)

        # Multi-channel: convolve each channel independently
        channels = [self._convolve_channel(image[:, :, c], kernel) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)

    @staticmethod
    def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Convolve a single 2D channel with a kernel using reflect padding."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Reflect padding
        padded = np.pad(channel.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        h, w = channel.shape
        output = np.zeros((h, w), dtype=np.float64)

        # Sliding window convolution
        for i in range(h):
            for j in range(w):
                region = padded[i:i + kh, j:j + kw]
                output[i, j] = np.sum(region * kernel)

        return np.clip(output, 0, 255).astype(np.uint8)
