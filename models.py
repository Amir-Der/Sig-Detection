from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    has_model = db.Column(db.Boolean, default=False)

class Signature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128))
    type = db.Column(db.String(10))  # 'real' or 'fake'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
