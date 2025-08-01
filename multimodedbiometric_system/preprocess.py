import cv2
import numpy as np
from skimage import restoration, img_as_ubyte

def load_image(path: str, as_gray: bool = False) -> np.ndarray:
    """
    Load an image from disk.
    :param path: Path to the image file.
    :param as_gray: If True, load as grayscale.
    :return: Image as a NumPy array (BGR if color).
    """
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {path}")
    return img

def apply_gaussian_blur(img: np.ndarray, ksize: tuple = (5,5), sigmaX: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise and detail.
    :param img: Input image.
    :param ksize: Kernel size; must be odd numbers.
    :param sigmaX: Gaussian kernel standard deviation in X.
    :return: Blurred image.
    """
    return cv2.GaussianBlur(img, ksize, sigmaX)

def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    Sharpen an image by applying a high-boost filter.
    :param img: Input image (grayscale or BGR).
    :return: Sharpened image.
    """
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)

def denoise_image_cv(img: np.ndarray, h: float = 10) -> np.ndarray:
    """
    Denoise a color image using OpenCV's Non-Local Means algorithm.
    :param img: Input color image (BGR).
    :param h: Parameter regulating filter strength for luminance component.
    :return: Denoised image.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize=7, searchWindowSize=21)

def denoise_image_sk(img: np.ndarray, method: str = 'bilateral') -> np.ndarray:
    """
    Denoise using scikit-image methods.
    :param img: Input image as uint8 or float in [0,1].
    :param method: 'bilateral' or 'wavelet'.
    :return: Denoised image (uint8).
    """
    if img.dtype != np.float32:
        img_f = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    else:
        img_f = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if method == 'bilateral':
        den = restoration.denoise_bilateral(img_f, multichannel=True)
    elif method == 'wavelet':
        den = restoration.denoise_wavelet(img_f, multichannel=True, convert2ycbcr=True)
    else:
        raise ValueError("Unsupported method: choose 'bilateral' or 'wavelet'")
    den_uint8 = img_as_ubyte(den)
    return cv2.cvtColor(den_uint8, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = load_image(img_path)

    blurred = apply_gaussian_blur(img, ksize=(7,7), sigmaX=1.5)
    sharpened = sharpen_image(blurred)
    denoised = denoise_image_cv(sharpened, h=15)

    cv2.imwrite("output_blurred.jpg", blurred)
    cv2.imwrite("output_sharpened.jpg", sharpened)
    cv2.imwrite("output_denoised.jpg", denoised)

    print("Preprocessing complete. Outputs:")
    print(" • output_blurred.jpg")
    print(" • output_sharpened.jpg")
    print(" • output_denoised.jpg")
