from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Signature
from import_cv2 import preprocess_image, extract_hog_feature
from train import train_user_model, predict_signature
import os, uuid, joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REMEMBER_COOKIE_DURATION'] = 3600 * 24 * 7

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        if User.query.filter_by(username=username).first():
            flash("کاربری با این نام وجود دارد")
            return redirect(url_for("register"))
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        login_user(user, remember=True)
        return redirect(url_for("dashboard"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user, remember=True)
            return redirect(url_for("dashboard"))
        flash("اطلاعات ورود اشتباه است")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = str(uuid.uuid4()) + ".png"
            folder = os.path.join(app.config["UPLOAD_FOLDER"], str(current_user.id))
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)
            file.save(filepath)
            result = predict_signature(current_user.id, filepath)
            os.remove(filepath)
    return render_template("dashboard.html", result=result)


@app.route("/upload/<sig_type>", methods=["POST"])
@login_required
def upload_signature(sig_type):
    files = request.files.getlist("files")
    if files and sig_type in ["real", "fake"]:
        folder = os.path.join(app.config["UPLOAD_FOLDER"], str(current_user.id), sig_type)
        os.makedirs(folder, exist_ok=True)

        for file in files:
            filename = str(uuid.uuid4()) + ".png"
            path = os.path.join(folder, filename)
            file.save(path)
            sig = Signature(filename=filename, type=sig_type, user_id=current_user.id)
            db.session.add(sig)

        db.session.commit()
        return "OK"
    return "Failed"


@app.route("/train", methods=["POST"])
@login_required
def train():
    model_path = train_user_model(current_user.id)
    if model_path:
        user = User.query.get(current_user.id)
        user.has_model = True
        db.session.commit()
        return jsonify({"status": "done"})
    return jsonify({"status": "fail"})
