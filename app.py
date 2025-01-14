import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_csv('Classification.csv')

label_encoders = {}
for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

X = data.drop('Drug', axis=1)
y = data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

try:
    model = joblib.load('xgb_model.pkl')
except FileNotFoundError:
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    joblib.dump(model, 'xgb_model.pkl')

st.set_page_config(page_title="Klasifikasi Obat dengan XGBoost", layout="wide")

st.title('🩺 Prediksi Kategori Obat dengan XGBoost')
st.info('Aplikasi ini memprediksi kategori obat berdasarkan fitur-fitur input menggunakan model XGBoost.')

with st.sidebar:
    st.header('Input Data Pasien')
    age = st.slider("Umur", int(data['Age'].min()), int(data['Age'].max()), int(data['Age'].mean()))
    na_to_k = st.slider("Rasio Na ke K", float(data['Na_to_K'].min()), float(data['Na_to_K'].max()), float(data['Na_to_K'].mean()))
    sex = st.selectbox("Jenis Kelamin", label_encoders['Sex'].classes_)
    bp = st.selectbox("Tekanan Darah", label_encoders['BP'].classes_)
    cholesterol = st.selectbox("Kolesterol", label_encoders['Cholesterol'].classes_)

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [label_encoders['Sex'].transform([sex])[0]],
        'BP': [label_encoders['BP'].transform([bp])[0]],
        'Cholesterol': [label_encoders['Cholesterol'].transform([cholesterol])[0]],
        'Na_to_K': [na_to_k]
    })

    st.markdown("---")
    st.markdown("**Aplikasi ini dibuat oleh**")
    st.success("Muhamad Reynaldi")

with st.expander("📊 Data dan Visualisasi"):
    st.subheader("Dataset")
    st.dataframe(data)

    st.subheader("Visualisasi Distribusi Kategori Obat")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Drug', data=data, palette='Set2', ax=ax)
    ax.set_title('Distribusi Kategori Obat', fontsize=16)
    ax.set_xlabel('Kategori Obat', fontsize=14)
    ax.set_ylabel('Jumlah', fontsize=14)
    st.pyplot(fig)

with st.expander("⚙️ Persiapan Data"):
    st.subheader("Statistik Dataset")
    st.write(data.describe())

    st.subheader("Distribusi Fitur")
    st.bar_chart(data.drop('Drug', axis=1))

prediction = model.predict(input_data)[0]
prediction_probabilities = model.predict_proba(input_data)[0]
prediction_label = label_encoders['Drug'].inverse_transform([prediction])[0]

df_prediction_proba = pd.DataFrame(
    [prediction_probabilities],
    columns=label_encoders['Drug'].classes_
)

st.subheader('📋 Hasil Prediksi Kategori Obat')
st.markdown(f"### Prediksi : ")
st.success(prediction_label)

st.dataframe(
    df_prediction_proba,
    column_config={
        drug: st.column_config.ProgressColumn(
            drug,
            format='%.3f',
            width='medium',
            min_value=0,
            max_value=1
        )
        for drug in label_encoders['Drug'].classes_
    },
    hide_index=True
)

with st.expander("⚙️ Evaluasi Model"):
    st.markdown("---")
    y_combined_pred = model.predict(X_test)
    accuracy_combined = accuracy_score(y_test, y_combined_pred)
    st.markdown(f"### Akurasi Model: **{accuracy_combined * 100:.2f}%**")

    st.markdown("### 📈 Laporan Klasifikasi")
    classification_dict_dynamic = classification_report(y_test, y_combined_pred, output_dict=True)

    for label, metrics in classification_dict_dynamic.items():
        if isinstance(metrics, dict):
            st.markdown(f"#### Kelas: **{label}**")
            df_metrics = pd.DataFrame(metrics, index=[0]).T.reset_index()
            df_metrics.columns = ['Metrik', 'Nilai']
            st.dataframe(df_metrics, use_container_width=True)
