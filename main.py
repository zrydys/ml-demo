# ml_census_income_demo/main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request

app = Flask(__name__)

def load_and_prepare_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = pd.read_csv(url, names=column_names, sep=',\s*', engine='python')
    data = data.replace('?', pd.NA).dropna()
    label_encoders = {}
    for col in data.select_dtypes(include='object'):
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    return data, label_encoders

data, label_encoders = load_and_prepare_data()
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html', feature_names=X.columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form.get(field)) for field in X.columns]
        prediction = model.predict([input_data])[0]
        label = label_encoders['income'].inverse_transform([prediction])[0] if 'income' in label_encoders else prediction
        return render_template('index.html', prediction_text=f'Predicted Income Class: {label}', feature_names=X.columns)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}', feature_names=X.columns)

if __name__ == '__main__':
    app.run(debug=True)

