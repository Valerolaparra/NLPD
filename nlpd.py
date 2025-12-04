"""
Normalised Laplacian Pyramid implemented in NumPy
Based on the PyTorch implementation by Alex Hepburn <ah13558@bristol.ac.uk>
NumPy adaptation: Non-differentiable version for forward-pass only
"""

import numpy as np
from scipy.ndimage import convolve, zoom
from typing import List, Tuple
import math


LAPLACIAN_FILTER = np.array([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                             [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                             [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]],
                            dtype=np.float32)


def pad_reflect(image: np.ndarray, padding: List[int]) -> np.ndarray:
    """
    Apply reflection padding to image.
    
    Args:
        image: Input image of shape (C, H, W) or (H, W)
        padding: [pad_left, pad_right, pad_top, pad_bottom]
    
    Returns:
        Padded image
    """
    pad_left, pad_right, pad_top, pad_bottom = padding
    
    if image.ndim == 3:
        # (C, H, W)
        return np.pad(image, 
                     ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                     mode='reflect')
    else:
        # (H, W)
        return np.pad(image,
                     ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode='reflect')


def pad_calc(im_size: List[int], filt_size: int, stride: int) -> List[int]:
    """
    Returns the amount of padding needed on [height, width] to maintain image
    size after convolution.
    
    Parameters
    ----------
    im_size : List[int]
        List of [height, width] of the image to pad.
    filt_size : int
        The width of the filter being used in the convolution, assumed to be
        square.
    stride : int
        Amount of stride in the convolution.
    
    Returns
    -------
    padding : List[int]
        Amount of padding needed [pad_left, pad_right, pad_top, pad_bottom]
    """
    out_height = math.ceil(float(im_size[0]) / float(stride))
    out_width = math.ceil(float(im_size[1]) / float(stride))
    pad_h = max((out_height - 1) * stride + filt_size - im_size[0], 0)
    pad_w = max((out_width - 1) * stride + filt_size - im_size[1], 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return [pad_left, pad_right, pad_top, pad_bottom]


def conv2d_grouped(image: np.ndarray, kernel: np.ndarray, stride: int = 1, 
                   groups: int = 1) -> np.ndarray:
    """
    Grouped 2D convolution similar to PyTorch's conv2d with groups.
    
    Args:
        image: Input of shape (C, H, W)
        kernel: Filter of shape (C, 1, kH, kW) for grouped convolution
        stride: Convolution stride
        groups: Number of groups (typically equals C for depthwise conv)
    
    Returns:
        Convolved output of shape (C, H_out, W_out)
    """
    C, H, W = image.shape
    out_channels, _, kH, kW = kernel.shape
    
    # Calculate output dimensions
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    
    output = np.zeros((out_channels, H_out, W_out), dtype=np.float32)
    
    if groups == C:
        # Depthwise convolution: each channel convolved with its own filter
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    patch = image[c, h_start:h_start+kH, w_start:w_start+kW]
                    output[c, i, j] = np.sum(patch * kernel[c, 0, :, :])
    else:
        # Standard grouped convolution
        channels_per_group = C // groups
        for g in range(groups):
            for c in range(channels_per_group):
                out_c = g * channels_per_group + c
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride
                        for in_c in range(channels_per_group):
                            in_channel = g * channels_per_group + in_c
                            patch = image[in_channel, h_start:h_start+kH, w_start:w_start+kW]
                            output[out_c, i, j] += np.sum(patch * kernel[out_c, in_c, :, :])
    
    return output


def interpolate_bilinear(image: np.ndarray, target_size: List[int]) -> np.ndarray:
    """
    Bilinear interpolation to resize image.
    
    Args:
        image: Input of shape (C, H, W)
        target_size: [target_H, target_W]
    
    Returns:
        Resized image of shape (C, target_H, target_W)
    """
    C, H, W = image.shape
    target_H, target_W = target_size
    
    # Calculate zoom factors
    zoom_factors = [1.0, target_H / H, target_W / W]
    
    # Use scipy's zoom with order=1 for bilinear interpolation
    return zoom(image, zoom_factors, order=1)


class LaplacianPyramid:
    """
    Normalised Laplacian Pyramid with fixed divisive normalisation filters.
    
    This is the NumPy version of the PyTorch LaplacianPyramid class.
    """
    
    def __init__(self, k: int, dims: int = 3, filt: np.ndarray = None):
        """
        Initialize Laplacian Pyramid.
        
        Args:
            k: Number of pyramid levels
            dims: Number of channels (default 3 for RGB)
            filt: Custom filter (if None, uses LAPLACIAN_FILTER)
        """
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (dims, 1, 1)),
                            (dims, 1, 5, 5))
        self.k = k
        self.dims = dims
        self.filt = filt.astype(np.float32)
        self.dn_filts, self.sigmas = self._dn_filters()
    
    def _dn_filters(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Create divisive normalisation filters for each pyramid level.
        
        Returns:
            Tuple of (dn_filters, sigmas)
        """
        sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
        dn_filts = []
        
        dn_filts.append(np.reshape([[0, 0.1011, 0],
                                    [0.1493, 0, 0.1460],
                                    [0, 0.1015, 0.]] * self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32))
        
        dn_filts.append(np.reshape([[0, 0.0757, 0],
                                    [0.1986, 0, 0.1846],
                                    [0, 0.0837, 0]] * self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32))
        
        dn_filts.append(np.reshape([[0, 0.0477, 0],
                                    [0.2138, 0, 0.2243],
                                    [0, 0.0467, 0]] * self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32))
        
        dn_filts.append(np.reshape([[0, 0, 0],
                                    [0.2503, 0, 0.2616],
                                    [0, 0, 0]] * self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32))
        
        dn_filts.append(np.reshape([[0, 0, 0],
                                    [0.2598, 0, 0.2552],
                                    [0, 0, 0]] * self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32))
        
        dn_filts.append(np.reshape([[0, 0, 0],
                                    [0.2215, 0, 0.0717],
                                    [0, 0, 0]] * self.dims,
                                   (self.dims, 1, 3, 3)).astype(np.float32))
        
        return dn_filts, sigmas
    
    def pyramid(self, im: np.ndarray) -> List[np.ndarray]:
        """
        Build normalised Laplacian pyramid.
        
        Args:
            im: Input image of shape (C, H, W) or (H, W, C)
                If (H, W, C), it will be converted to (C, H, W)
        
        Returns:
            List of pyramid levels (normalised Laplacian bands)
        """
        # Ensure image is (C, H, W)
        if im.ndim == 3 and im.shape[2] == self.dims:
            # Convert from (H, W, C) to (C, H, W)
            im = np.transpose(im, (2, 0, 1))
        
        J = im.astype(np.float32)
        pyr = []
        
        for i in range(self.k):
            # Downsample: convolve with stride 2
            J_padding_amount = pad_calc([J.shape[1], J.shape[2]], 
                                       self.filt.shape[2], stride=2)
            J_padded = pad_reflect(J, J_padding_amount)
            I = conv2d_grouped(J_padded, self.filt, stride=2, groups=self.dims)
            
            # Upsample back to original size
            I_up = interpolate_bilinear(I, [J.shape[1], J.shape[2]])
            
            # Convolve upsampled image
            I_padding_amount = pad_calc([I_up.shape[1], I_up.shape[2]], 
                                       self.filt.shape[2], stride=1)
            I_up_padded = pad_reflect(I_up, I_padding_amount)
            I_up_conv = conv2d_grouped(I_up_padded, self.filt, stride=1, 
                                      groups=self.dims)
            
            # Laplacian band (high frequency details)
            out = J - I_up_conv
            
            # Divisive normalisation
            out_padding_amount = pad_calc([out.shape[1], out.shape[2]], 
                                         self.dn_filts[i].shape[2], stride=1)
            out_abs_padded = pad_reflect(np.abs(out), out_padding_amount)
            out_conv = conv2d_grouped(out_abs_padded, self.dn_filts[i], 
                                     stride=1, groups=self.dims)
            
            # Normalise
            out_norm = out / (self.sigmas[i] + out_conv)
            pyr.append(out_norm)
            
            # Move to next level
            J = I
        
        return pyr
    
    def compare(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compare two images using normalised Laplacian pyramid distance.
        
        Args:
            x1: First image of shape (C, H, W) or (H, W, C)
            x2: Second image of shape (C, H, W) or (H, W, C)
        
        Returns:
            Distance metric (L0.6 norm of per-level MSE)
        """
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        
        total = []
        # Calculate difference in perceptual space
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = np.sqrt(np.mean(diff))
            total.append(sqrt)
        
        # L0.6 norm
        return np.linalg.norm(total, ord=0.6)


class GDN:
    """
    Generalised Divisive Normalisation (GDN) - NumPy version.
    
    Simplified non-trainable version for forward pass only.
    """
    
    def __init__(self, n_channels: int, kernel_size: int = 1, stride: int = 1,
                 padding: int = 0, gamma_init: float = 0.1,
                 reparam_offset: float = 2**-18, beta_min: float = 1e-6,
                 apply_independently: bool = False):
        """
        Initialize GDN layer.
        
        Args:
            n_channels: Number of input channels
            kernel_size: Size of kernel for spatial GDN
            stride: Stride for convolution
            padding: Padding for convolution
            gamma_init: Initial value for gamma weights
            reparam_offset: Reparameterisation offset
            beta_min: Minimum beta value
            apply_independently: Whether to apply channel-wise independently
        """
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.reparam_offset = reparam_offset
        self.beta_reparam = (beta_min + reparam_offset**2)**0.5
        
        if apply_independently:
            self.groups = n_channels
        else:
            self.groups = 1
        
        # Initialize gamma (identity-like matrix)
        gamma = np.eye(n_channels, dtype=np.float32)
        gamma = gamma.reshape(n_channels, n_channels, 1, 1)
        gamma = np.tile(gamma, (1, 1, kernel_size, kernel_size))
        gamma = np.sqrt(gamma_init * gamma + reparam_offset**2)
        gamma = gamma * gamma
        
        if apply_independently:
            # Extract diagonal for independent application
            gammas = [gamma[i, i:i+1, :, :] for i in range(n_channels)]
            gamma = np.stack(gammas, axis=0)
        
        self.gamma = gamma
        
        # Initialize beta
        beta = np.ones((n_channels,), dtype=np.float32)
        beta = np.sqrt(beta + reparam_offset**2)
        self.beta = beta
    
    def _clamp_parameters(self):
        """Clamp gamma and beta to valid ranges."""
        self.gamma = np.maximum(self.gamma, self.reparam_offset)
        self.beta = np.maximum(self.beta, self.beta_reparam)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of GDN layer.
        
        Args:
            x: Input of shape (C, H, W)
        
        Returns:
            Normalised output of shape (C, H, W)
        """
        self._clamp_parameters()
        
        # Compute normalisation pool: conv(x^2)
        x_squared = x * x
        
        # Add padding if needed
        if self.padding > 0:
            pad_amount = [self.padding] * 4
            x_squared = pad_reflect(x_squared, pad_amount)
        
        # Convolve
        norm_pool = conv2d_grouped(x_squared, self.gamma, stride=self.stride, 
                                  groups=self.groups)
        
        # Add beta bias
        for c in range(self.n_channels):
            norm_pool[c] += self.beta[c]
        
        # Square root
        norm_pool = np.sqrt(norm_pool)
        
        # Interpolate to match input size if needed
        if norm_pool.shape[1:] != x.shape[1:]:
            norm_pool = interpolate_bilinear(norm_pool, [x.shape[1], x.shape[2]])
        
        # Divide
        output = x / norm_pool
        return output


class LaplacianPyramidGDN:
    """
    Normalised Laplacian Pyramid with learnable GDN normalisation.
    
    NumPy version with fixed (non-trainable) GDN parameters.
    """
    
    def __init__(self, k: int, dims: int = 3, filt: np.ndarray = None):
        """
        Initialize Laplacian Pyramid with GDN.
        
        Args:
            k: Number of pyramid levels
            dims: Number of channels (default 3 for RGB)
            filt: Custom filter (if None, uses LAPLACIAN_FILTER)
        """
        if filt is None:
            filt = np.reshape(np.tile(LAPLACIAN_FILTER, (dims, 1, 1)),
                            (dims, 1, 5, 5))
        self.k = k
        self.dims = dims
        self.filt = filt.astype(np.float32)
        self.gdns = [GDN(dims, apply_independently=True) for _ in range(k)]
    
    def pyramid(self, im: np.ndarray) -> List[np.ndarray]:
        """
        Build normalised Laplacian pyramid with GDN.
        
        Args:
            im: Input image of shape (C, H, W) or (H, W, C)
        
        Returns:
            List of pyramid levels (GDN-normalised Laplacian bands)
        """
        # Ensure image is (C, H, W)
        if im.ndim == 3 and im.shape[2] == self.dims:
            im = np.transpose(im, (2, 0, 1))
        
        J = im.astype(np.float32)
        pyr = []
        
        for i in range(self.k):
            # Pad and downsample
            J_padded = pad_reflect(J, [2, 2, 2, 2])
            I = conv2d_grouped(J_padded, self.filt, stride=2, groups=self.dims)
            
            # Upsample
            I_up = interpolate_bilinear(I, [J.shape[1] * 2, J.shape[2] * 2])
            
            # Convolve upsampled
            I_up_padded = pad_reflect(I_up, [2, 2, 2, 2])
            I_up_conv = conv2d_grouped(I_up_padded, self.filt, stride=1, 
                                      groups=self.dims)
            
            # Match sizes if needed
            if J.shape != I_up_conv.shape:
                I_up_conv = interpolate_bilinear(I_up_conv, [J.shape[1], J.shape[2]])
            
            # Laplacian with GDN normalisation
            laplacian = J - I_up_conv
            pyr.append(self.gdns[i].forward(laplacian))
            
            J = I
        
        return pyr
    
    def compare(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compare two images using GDN-normalised Laplacian pyramid.
        
        Args:
            x1: First image of shape (C, H, W) or (H, W, C)
            x2: Second image of shape (C, H, W) or (H, W, C)
        
        Returns:
            Distance metric (L0.6 norm of per-level MSE)
        """
        y1 = self.pyramid(x1)
        y2 = self.pyramid(x2)
        
        total = []
        for z1, z2 in zip(y1, y2):
            diff = (z1 - z2) ** 2
            sqrt = np.sqrt(np.mean(diff))
            total.append(sqrt)
        
        return np.linalg.norm(total, ord=0.6)


# Example usage
if __name__ == "__main__":
    print("=== Normalised Laplacian Pyramid - NumPy Implementation ===\n")
    
    # Create a test image (RGB)
    height, width = 256, 256
    test_image = np.random.rand(3, height, width).astype(np.float32)
    
    print("Testing LaplacianPyramid...")
    lap_pyr = LaplacianPyramid(k=4, dims=3)
    pyramid_levels = lap_pyr.pyramid(test_image)
    print(f"Created {len(pyramid_levels)} pyramid levels")
    for i, level in enumerate(pyramid_levels):
        print(f"  Level {i}: shape {level.shape}")
    
    # Test comparison
    test_image2 = test_image + np.random.randn(*test_image.shape) * 0.1
    distance = lap_pyr.compare(test_image, test_image2)
    print(f"\nDistance between images: {distance:.6f}")
    
    print("\n" + "="*60)
    print("Testing LaplacianPyramidGDN...")
    lap_pyr_gdn = LaplacianPyramidGDN(k=4, dims=3)
    pyramid_levels_gdn = lap_pyr_gdn.pyramid(test_image)
    print(f"Created {len(pyramid_levels_gdn)} pyramid levels with GDN")
    for i, level in enumerate(pyramid_levels_gdn):
        print(f"  Level {i}: shape {level.shape}")
    
    distance_gdn = lap_pyr_gdn.compare(test_image, test_image2)
    print(f"\nGDN Distance between images: {distance_gdn:.6f}")