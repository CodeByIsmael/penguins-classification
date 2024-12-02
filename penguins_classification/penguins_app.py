from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('models/log_reg_model.pkl', 'rb') as f:
    dv_log_reg, scaler_log_reg, le_log_reg, log_reg_model = pickle.load(f)

with open('models/svm_model.pkl', 'rb') as f:
    dv_svm, scaler_svm, le_svm, svm_model = pickle.load(f)

with open('models/dt_model.pkl', 'rb') as f:
    dv_dt, scaler_dt, le_dt, dt_model = pickle.load(f)

with open('models/knn_model.pkl', 'rb') as f:
    dv_knn, scaler_knn, le_knn, knn_model = pickle.load(f)


def preprocess(data, dv, scaler):
    data_cat = dv.transform([data])
    data_num = scaler.transform([[
        data['bill_length_mm'],
        data['bill_depth_mm'],
        data['flipper_length_mm'],
        data['body_mass_g']
    ]])
    data_prepared = np.hstack((data_num, data_cat))
    return data_prepared

# Endpoint para Regresión Logística
@app.route('/predict/log_reg', methods=['POST'])
def predict_log_reg():
    data = request.get_json()
    data_prepared = preprocess(data, dv_log_reg, scaler_log_reg)
    pred_proba = log_reg_model.predict_proba(data_prepared)[0]
    pred_class = log_reg_model.predict(data_prepared)[0]
    species = le_log_reg.inverse_transform([pred_class])[0]
    return jsonify({
        'species': species,
        'probability': float(np.max(pred_proba))
    })

# Endpoint para SVM
@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    data = request.get_json()
    data_prepared = preprocess(data, dv_svm, scaler_svm)
    pred_proba = svm_model.predict_proba(data_prepared)[0]
    pred_class = svm_model.predict(data_prepared)[0]
    species = le_svm.inverse_transform([pred_class])[0]
    return jsonify({
        'species': species,
        'probability': float(np.max(pred_proba))
    })

# Endpoint para Árbol de Decisión
@app.route('/predict/dt', methods=['POST'])
def predict_dt():
    data = request.get_json()
    data_prepared = preprocess(data, dv_dt, scaler_dt)
    pred_proba = dt_model.predict_proba(data_prepared)[0]
    pred_class = dt_model.predict(data_prepared)[0]
    species = le_dt.inverse_transform([pred_class])[0]
    return jsonify({
        'species': species,
        'probability': float(np.max(pred_proba))
    })

# Endpoint para KNN
@app.route('/predict/knn', methods=['POST'])
def predict_knn():
    data = request.get_json()
    data_prepared = preprocess(data, dv_knn, scaler_knn)
    pred_proba = knn_model.predict_proba(data_prepared)[0]
    pred_class = knn_model.predict(data_prepared)[0]
    species = le_knn.inverse_transform([pred_class])[0]
    return jsonify({
        'species': species,
        'probability': float(np.max(pred_proba))
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)