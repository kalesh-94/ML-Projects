import pickle
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, redirect

# Directly specify the model file path
model_path = 'model.pkl' ## give the path of your model.pkl file

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        
        features = [
            int(request.form['pregnancies']),
            int(request.form['glucose']),
            int(request.form['bp']),
            int(request.form['skinthickness']),
            int(request.form['insulin']),
            int(request.form['bmi']),
            int(request.form['diabetespedigreefunction']),
            int(request.form['age'])
        ]
        prediction = model.predict([features])[0]
        result = 'Yes' if prediction == 1 else 'No'
        return render_template('output.html', output=result)
    else:
        return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
