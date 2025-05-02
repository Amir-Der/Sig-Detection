from flask import Flask, render_template, request, jsonify
from import_cv2 import preprocess_image, extract_hog_feature
from train import train_temp_model, predict_with_temp_model
import os
import uuid

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def dashboard():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return jsonify(result="فایلی ارسال نشده")
        img = preprocess_image(file.stream)
        feature = extract_hog_feature(img).reshape(1, -1)
        result = predict_with_temp_model(feature)
        return jsonify(result=result)
    return render_template("dashboard.html")


@app.route("/train", methods=["POST"])
def train():
    real_files = request.files.getlist("real_files")
    fake_files = request.files.getlist("fake_files")

    if len(real_files) < 1 or len(fake_files) < 1:
        return jsonify(result="حداقل یک فایل واقعی و یک فایل جعلی نیاز است")

    try:
        X, y = [], []
        for f in real_files:
            img = preprocess_image(f.stream)
            X.append(extract_hog_feature(img))
            y.append(1)
        for f in fake_files:
            img = preprocess_image(f.stream)
            X.append(extract_hog_feature(img))
            y.append(0)

        train_temp_model(X, y)
        return jsonify(result="✅ مدل با موفقیت آموزش داده شد.")
    except Exception as e:
        return jsonify(result=f"⛔ خطا در آموزش مدل: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
