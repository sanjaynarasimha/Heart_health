from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the logistic regression model
model_path = 'final-model.pkl'
model = joblib.load(model_path)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from form
        form_data = request.form
        features = [float(form_data['cp']), float(form_data['ca']), float(form_data['thalach']),
                    float(form_data['oldpeak']), float(form_data['age']), float(form_data['thal']),
                    float(form_data['trestbps']), float(form_data['chol']), float(form_data['exang']),
                    float(form_data['slope'])]
        features = np.array(features).reshape(1, -1)

        # Check model type and predict
        if hasattr(model, 'predict'):
            prediction = model.predict(features)
            # Render prediction result
            return render_template('result.html', prediction=int(prediction[0]))
        else:
            return "Model loaded is not a scikit-learn model with a predict method."

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
