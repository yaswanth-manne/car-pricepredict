from flask import Flask, render_template,request
import pandas as pd
import pickle
import numpy as np
app=Flask(__name__)

model = pickle.load(open("random_forest.pkl",'rb'))
car = pd.read_csv("cats_data.csv") 


@app.route('/')
def index():
    companies = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel = sorted(car['fuel'].unique())
    return render_template('index.html',companies=companies, years=year,fuels=fuel)

@app.route('/predict',methods=['POST'])
def predict():
    company= request.form.get("company")
    year= int(request.form.get("year"))
    fuel= request.form.get("fuel")
    km_driven= int(request.form.get("km_driven"))
    

    prediction = model.predict(pd.DataFrame([[company, year, km_driven, fuel]], columns=['name', 'year', 'km_driven', 'fuel']))
    print(prediction[0])
    return str(np.round(prediction[0],2))
if __name__=="__main__":
    app.run(debug=True)