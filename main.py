import os

import numpy as np
from PIL import Image


def load_image(
        image_path: str
)->np.ndarray:
    return np.array(Image.open(image_path))


def calculate_zero_pad(
        kernel_size: int
)-> int:
    pad_width = (kernel_size - 1)//2
    return pad_width


def manual_zero_pad(
        array: np.ndarray,
        pad_width: int,
        pad_height: int
)-> int:

    padded_array = np.zeros(
        (array.shape[0] + 2 * pad_height,
         array.shape[1] + 2 * pad_width),
        dtype=np.float32
    )
    padded_array[pad_height:-pad_height, pad_width:-pad_width] = array
    return padded_array


def apply_convolution(
        image: np.ndarray,
        kernel: np.ndarray
)->np.ndarray:

    image = image.astype(np.float32)
    kernel = kernel.astype(np.float32)

    kernel_height, kernel_width = kernel.shape

    pad_width = calculate_zero_pad(kernel_width)
    pad_height = calculate_zero_pad(kernel_height)
    print(50 * "-", "\nManual zero padding width:", pad_width)
    print(50 * "-", "\nManual zero padding height:", pad_height)

    zero_padded_image = manual_zero_pad(
        array=image,
        pad_width=pad_width,
        pad_height=pad_height
        )
    print(50 * "-", "\nManual zero padded image:\n", zero_padded_image)

    new_image_array = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = zero_padded_image[i:i+kernel_height, j:j+kernel_width]
            new_image_array[i, j] = np.sum(roi * kernel)
    return new_image_array.astype(np.uint8)


def save_image(
        image_array: np.ndarray,
        filename: str
) -> None:
    Image.fromarray(image_array).save(filename)
    return print(50*'-',f"\nImage saved to {filename}")


def normalize_kernel(kernel):
    return kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel


def apply_thresholding(
        image_array: np.ndarray,
        threshold_value: np.float32
) -> np.ndarray:
    return np.where(image_array > threshold_value, 1, 0).astype(np.bool)


def main():
    base_image_path = "Images"
    if not os.path.exists(base_image_path):
        os.mkdir(base_image_path)

    image_array = load_image(
        image_path=os.path.join(base_image_path, "Hubble Space Telescope.tif")
    )
    print(50 * "-", "\nOriginal Image Array:\n", image_array)
    print(50*"-","\nOriginal Image shape:",image_array.shape)

    kernel = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    conv_matrix = normalize_kernel(kernel)

    print(50 * "-", "\nConv. Kernel:\n", conv_matrix)
    print(50 * "-", "\nConv. Kernel Shape:", conv_matrix.shape)

    output_image = apply_convolution(
        image=image_array,
        kernel=conv_matrix
    )
    print(50 * "-", "\nOutput Image:\n", output_image)
    print(50 * "-", "\nOutput Image Shape:", output_image.shape)
    save_image(
        image_array=output_image,
        filename=os.path.join(base_image_path, "applied_conv_image.png")
    )

    threshold_image = apply_thresholding(
        image_array=output_image,
        threshold_value=np.mean(output_image) + 30
    )
    print(50 * "-", "\nOutput Image(Thresholding):\n", threshold_image)
    save_image(
        image_array=threshold_image,
        filename=os.path.join(base_image_path, "applied_thresholding_image.png")
    )

if __name__ == '__main__':
    main()
