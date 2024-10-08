from flask import Flask, redirect, request, render_template, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from email_validator import validate_email, EmailNotValidError
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secret key for session management

# Configure the database connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/crop_recommendation'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Define the Prediction model
class Prediction(db.Model):
    __tablename__ = 'prediction'
    
    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    predicted_crop = db.Column(db.String(50), nullable=False)

# Define the Contact model
class Contact(db.Model):
    __tablename__ = 'contact'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)

# Load the model and scalers
model = pickle.load(open('D:/Abhinav/Test/Digital_Agronomy/model/model.pkl', 'rb'))
sc = pickle.load(open('D:/Abhinav/Test/Digital_Agronomy/modelstandscaler.pkl', 'rb'))
ms = pickle.load(open('D:/Abhinav/Test/Digital_Agronomy/modelminmaxscaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/products')
def products():
    return render_template('products.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        # Validate email
        try:
            validate_email(email)
        except EmailNotValidError as e:
            flash(f'Invalid email address: {e}', 'danger')
            return render_template('contact.html')

        new_contact = Contact(name=name, email=email, subject=subject, message=message)
        db.session.add(new_contact)
        db.session.commit()
        
        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username  # Store username in session
            flash('Login successful!', 'success')
            return redirect(url_for('index'))  # Redirect to the index route
        else:
            flash('Invalid Credentials. Please try again.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)  # Remove username from session
    flash('You have been logged out.', 'info')
    return redirect('/login')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['email']  # Updated to match the form field name
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validate email
        try:
            validate_email(username)
        except EmailNotValidError as e:
            error = str(e)
            return render_template('signup.html', error=error)

        if password != confirm_password:
            error = 'Passwords do not match.'
            return render_template('signup.html', error=error)
        
        if User.query.filter_by(username=username).first():
            error = 'Email already in use.'
            return render_template('signup.html', error=error)

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! You can now log in.', 'success')
        return redirect('/login')

    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        N = request.form.get('Nitrogen')
        P = request.form.get('Phosporus')
        K = request.form.get('Potassium')
        temp = request.form.get('Temperature')
        humidity = request.form.get('Humidity')
        ph = request.form.get('Ph')
        rainfall = request.form.get('Rainfall')

        if None in [N, P, K, temp, humidity, ph, rainfall]:
            flash("Error: Missing form field", 'danger')
            return redirect(url_for('predict'))

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

        new_prediction = Prediction(
            nitrogen=N,
            phosphorus=P,
            potassium=K,
            temperature=temp,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            predicted_crop=crop
        )
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('predict.html', result=result)
    else:
        return render_template('predict.html')

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
