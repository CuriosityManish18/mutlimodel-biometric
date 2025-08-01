import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from biometric_fusion import fuse_biometric_features

def load_data(csv_path, methods=None, weights=None):
    """
    Load CSV containing image paths and labels, and extract fused features.
    Expects columns: fingerprint_path, face_path, ear_path, label
    """
    df = pd.read_csv(csv_path)
    X, y = [], df['label'].values
    for _, row in df.iterrows():
        vect = fuse_biometric_features(
            row['fingerprint_path'],
            row['face_path'],
            row['ear_path'],
            methods=methods,
            weights=weights
        )
        X.append(vect)
    return np.vstack(X), y

def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP classifier on fused biometric features"
    )
    parser.add_argument(
        'csv', help="CSV file with columns: fingerprint_path,face_path,ear_path,label"
    )
    parser.add_argument('--test-size', type=float, default=0.2,
                        help="Fraction of data for testing (default: 0.2)")
    parser.add_argument('--random-state', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[100],
                        help="Sizes of hidden layers, e.g. --hidden-layers 128 64")
    parser.add_argument('--max-iter', type=int, default=200,
                        help="Maximum iterations for convergence")
    parser.add_argument('--model-out', default='mlp_model.pkl',
                        help="Path to save trained model and scaler")
    parser.add_argument('--weights', type=float, nargs=3, default=None,
                        help="Weights for finger, face, ear fusion (length 3)")
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['hog','orb','sift','color'],
                        help="Feature methods to use for each modality")
    args = parser.parse_args()

    print("Loading and extracting features...")
    X, y = load_data(args.csv, methods=args.methods, weights=args.weights)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        max_iter=args.max_iter,
        random_state=args.random_state
    )

    print("Training MLP classifier...")
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({'model': clf, 'scaler': scaler}, args.model_out)
    print(f"Saved model and scaler to {args.model_out}")

if __name__ == "__main__":
    main()
