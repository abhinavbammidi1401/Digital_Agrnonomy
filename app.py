from flask import Flask, redirect,request,render_template, url_for
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = pickle.load(open('D:/Abhinav/Test/CRSystem/model.pkl','rb'))
sc = pickle.load(open('D:/Abhinav/Test/CRSystem/standscaler.pkl','rb'))
ms = pickle.load(open('D:/Abhinav/Test/CRSystem/minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

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

@app.route('/contact')
def contact():
   return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add your authentication logic here
        #if username == 'admin' and password == 'password':  # Example condition
         #   return redirect(url_for('index'))
        #else:
         #   error = 'Invalid Credentials. Please try again.'
          #  return render_template('login.html', error=error)
    return render_template('login.html')

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

        # Check if any of the form fields are missing
        if None in [N, P, K, temp, humidity, ph, rainfall]:
            return "Error: Missing form field", 400

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
        return render_template('predict.html', result=result)
    else:
        return render_template('predict.html')


# python main
if __name__ == "__main__":
    app.run(debug=True)