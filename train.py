import os, joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from import_cv2 import preprocess_image, extract_hog_feature
from pathlib import Path

def load_user_data(user_id, base_folder="uploads"):
    real_path = Path(base_folder) / str(user_id) / "real"
    fake_path = Path(base_folder) / str(user_id) / "fake"
    real = [extract_hog_feature(preprocess_image(p)) for p in real_path.glob("*.png")]
    fake = [extract_hog_feature(preprocess_image(p)) for p in fake_path.glob("*.png")]
    X = np.array(real + fake, dtype=np.float32)
    y = np.array([1] * len(real) + [0] * len(fake))
    return X, y

def train_user_model(user_id):
    try:
        X, y = load_user_data(user_id)
        if len(X) < 10: return None
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(20, len(X), X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        model = SVC(C=10, gamma=0.001, probability=True)
        model.fit(X_pca, y)
        path = f"user_models/{user_id}.pkl"
        os.makedirs("user_models", exist_ok=True)
        joblib.dump((model, scaler, pca), path)
        return path
    except Exception:
        return None

def predict_signature_with_conf(user_id, img_path):
    try:
        model, scaler, pca = joblib.load(f"user_models/{user_id}.pkl")
        feature = extract_hog_feature(preprocess_image(img_path)).reshape(1, -1)
        feature = scaler.transform(feature)
        feature = pca.transform(feature)
        prob = model.predict_proba(feature)[0][1]
        pred = model.predict(feature)[0]
        label = "امضا معتبر است ✅" if pred == 1 else "امضا جعلی است ❌"
        return label, float(prob)
    except:
        return "⛔ مدل آموزش داده نشده یا خطا در پردازش تصویر", 0
