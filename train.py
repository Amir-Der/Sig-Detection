from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# مدل موقتی در حافظه
model = None
scaler = None
pca = None

def train_temp_model(X, y):
    global model, scaler, pca
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # نرمال‌سازی
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # کاهش ابعاد
    n_components = min(20, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # آموزش مدل
    model = SVC(C=10, gamma=0.001, probability=True)
    model.fit(X_pca, y)

def predict_with_temp_model(features):
    global model, scaler, pca
    if model is None:
        return "ابتدا مدل را آموزش دهید"
    features = scaler.transform(features)
    features = pca.transform(features)
    prediction = model.predict(features)[0]
    return "امضا معتبر است ✅" if prediction == 1 else "امضا جعلی است ❌"
