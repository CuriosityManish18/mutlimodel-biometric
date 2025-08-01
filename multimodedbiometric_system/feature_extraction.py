import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from preprocess import load_image

def extract_orb_features(img, nfeatures=500):
    """
    Extract ORB keypoints and descriptors.
    :param img: Input image (BGR or grayscale).
    :param nfeatures: Maximum number of ORB features.
    :return: keypoints, descriptors
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    orb = cv2.ORB_create(nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def extract_sift_features(img):
    """
    Extract SIFT keypoints and descriptors.
    Requires OpenCV contrib (opencv-contrib-python).
    :param img: Input image (BGR or grayscale).
    :return: keypoints, descriptors
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def extract_hog_features(img, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9, visualize=False):
    """
    Extract HOG (Histogram of Oriented Gradients) features.
    :param img: Input image (BGR or grayscale).
    :param pixels_per_cell: Size (in pixels) of a cell.
    :param cells_per_block: Number of cells in each block.
    :param orientations: Number of orientation bins.
    :param visualize: If True, returns (features, hog_image), else just features.
    :return: feature vector (and hog_image if visualize=True)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    features, hog_image = hog(
        gray, 
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    if visualize:
        # Rescale hog_image for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 255))
        return features, hog_image_rescaled
    return features

def extract_color_histogram(img, bins=(8, 8, 8)):
    """
    Compute a 3D color histogram in the BGR color space and normalize it.
    :param img: Input image (BGR).
    :param bins: Number of bins for each channel.
    :return: Flattened, normalized histogram vector.
    """
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = load_image(img_path)

    # ORB Features
    orb_kp, orb_desc = extract_orb_features(img)
    print(f"ORB: {len(orb_kp)} keypoints, descriptor shape: {orb_desc.shape if orb_desc is not None else None}")

    # SIFT Features
    try:
        sift_kp, sift_desc = extract_sift_features(img)
        print(f"SIFT: {len(sift_kp)} keypoints, descriptor shape: {sift_desc.shape if sift_desc is not None else None}")
    except Exception as e:
        print(f"SIFT extraction failed: {e}")

    # HOG Features and visualization
    hog_features, hog_img = extract_hog_features(img, visualize=True)
    print(f"HOG: feature vector length: {len(hog_features)}")
    cv2.imwrite("hog_visualization.jpg", hog_img)

    # Color Histogram
    hist = extract_color_histogram(img)
    print(f"Color histogram length: {len(hist)}")

    print("Feature extraction complete. HOG visualization saved as 'hog_visualization.jpg'.")
