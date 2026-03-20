from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    sqfootage = float(request.form['sqfootage'])
    bedroom = int(request.form['bedroom'])
    bathroom = int(request.form['bathroom'])
    location = request.form['location']

    location_city = 0
    location_rural = 0
    location_suburb = 0

    if location == "city":
        location_city = 1
    elif location == "rural":
        location_rural = 1
    else:
        location_suburb = 1

    user_data = pd.DataFrame({
        'sqfootage':[sqfootage],
        'bedroom':[bedroom],
        'bathroom':[bathroom],
        'location_city':[location_city],
        'location_rural':[location_rural],
        'location_suburb':[location_suburb]
    })

    prediction = model.predict(user_data)

    return render_template("index.html",
           prediction_text=f"Predicted House Price: ₹ {round(prediction[0])}")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)