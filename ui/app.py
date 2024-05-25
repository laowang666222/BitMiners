from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

from models import User, Data, ModelResult

# 创建数据库
with app.app_context():
    db.create_all()

from flask import render_template, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from forms import RegistrationForm, LoginForm
from models import User

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)


import os
from flask import request
from werkzeug.utils import secure_filename
from utils.data_processing import preprocess_data

UPLOAD_FOLDER = 'data/raw'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            preprocess_data(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded and processed', 'success')
            return redirect(url_for('index'))
    return render_template('upload.html')


from models import ModelResult
from utils.model_training import train_knn, train_decision_tree, train_svm, train_mlp

@app.route('/train_model', methods=['POST'])
def train_model():
    model_name = request.form['model_name']
    if model_name == 'knn':
        accuracy = train_knn()
    elif model_name == 'decision_tree':
        accuracy = train_decision_tree()
    elif model_name == 'svm':
        accuracy = train_svm()
    elif model_name == 'mlp':
        accuracy = train_mlp()

    new_result = ModelResult(model_name=model_name, accuracy=accuracy)
    db.session.add(new_result)
    db.session.commit()

    flash(f'{model_name} trained successfully with accuracy {accuracy}', 'success')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    results = ModelResult.query.all()
    return render_template('results.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
