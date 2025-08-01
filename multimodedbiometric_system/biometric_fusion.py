"""
Module for fusing biometric features from fingerprint, face, and ear images.
"""

import numpy as np
from preprocess import load_image
from feature_extraction import (
    extract_hog_features,
    extract_orb_features,
    extract_sift_features,
    extract_color_histogram
)

def feature_level_fusion(feature_vectors, weights=None):
    """
    Fuse multiple feature vectors by weighted concatenation.

    :param feature_vectors: List of 1D numpy arrays.
    :param weights: List of weights for each vector.
    :return: Fused feature vector as 1D numpy array.
    """
    if weights is None:
        weights = [1.0] * len(feature_vectors)
    if len(weights) != len(feature_vectors):
        raise ValueError("Number of weights must match number of feature vectors.")
    weighted = [w * fv for fv, w in zip(feature_vectors, weights)]
    return np.concatenate(weighted)

def score_level_fusion(scores, weights=None):
    """
    Fuse match scores by weighted sum rule.

    :param scores: List or array of scores.
    :param weights: List or array of weights.
    :return: Fused score (float).
    """
    scores = np.asarray(scores, dtype=float)
    if weights is None:
        weights = np.ones_like(scores)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(scores):
        raise ValueError("Number of weights must match number of scores.")
    return np.sum(weights * scores)

def extract_features_for_fusion(img_path, methods=('hog','orb','sift','color')):
    """
    Extract selected features from an image for fusion.

    :param img_path: Path to input image.
    :param methods: Tuple of feature methods: 'hog', 'orb', 'sift', 'color'.
    :return: 1D numpy array of concatenated features.
    """
    img = load_image(img_path, as_gray=False)
    feats = []
    if 'hog' in methods:
        hog_feat = extract_hog_features(img)
        feats.append(hog_feat)
    if 'orb' in methods:
        kp, desc = extract_orb_features(img)
        desc_vec = desc.flatten() if desc is not None else np.array([])
        feats.append(desc_vec)
    if 'sift' in methods:
        kp, desc = extract_sift_features(img)
        desc_vec = desc.flatten() if desc is not None else np.array([])
        feats.append(desc_vec)
    if 'color' in methods:
        color_hist = extract_color_histogram(img)
        feats.append(color_hist)
    if not feats:
        raise ValueError("No feature methods selected.")
    return np.concatenate(feats)

def fuse_biometric_features(finger_path, face_path, ear_path, methods=None, weights=None):
    """
    Extract and fuse features from finger, face, and ear images.

    :param finger_path: Path to fingerprint image.
    :param face_path: Path to face image.
    :param ear_path: Path to ear image.
    :param methods: List of feature methods to apply.
    :param weights: List of weights for finger, face, and ear fusion.
    :return: Fused feature vector.
    """
    if methods is None:
        methods = ['hog', 'orb', 'sift', 'color']
    # Extract per-modality feature vectors
    feat_finger = extract_features_for_fusion(finger_path, methods)
    feat_face   = extract_features_for_fusion(face_path, methods)
    feat_ear    = extract_features_for_fusion(ear_path, methods)
    # Fuse at feature level
    fused_vector = feature_level_fusion([feat_finger, feat_face, feat_ear], weights)
    return fused_vector

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python biometric_fusion.py <fingerprint> <face> <ear>")
        sys.exit(1)
    fused = fuse_biometric_features(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"Fused feature vector length: {fused.shape[0]}")
