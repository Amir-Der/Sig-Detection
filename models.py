from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import numpy as np

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Signature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(10), nullable=False)  # real یا fake
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
