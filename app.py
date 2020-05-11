from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('capita.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("capita.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
        int_features=[int(x) for x in request.form.values()]
        final=[np.array(int_features)]
        prediction=model.predict(final)
        output = round(prediction[0], 2)
        cof = model.coef_

        return render_template('capita.html',prediction_text="Income should be {}".format(output))

        return render_template('capita.html',cof_value="Coef value is {}".format(cof))




if __name__ == '__main__':
    app.run(debug=True)
