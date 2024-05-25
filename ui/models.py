from app import db
from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    registered_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    label = db.Column(db.Integer, nullable=False)

class ModelResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    created_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
