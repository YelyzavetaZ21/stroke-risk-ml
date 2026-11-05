import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / 'models' / 'lr_calibrated.joblib')
    with open(ROOT / 'models' / 'config.json') as f:
        config = json.load(f)
    return model, config


model, config = load_artifacts()

st.title('Stroke risk score')
st.warning('Educational project. Not a medical tool and not a diagnosis.')

age = st.number_input('Age (years)', min_value=0.08, max_value=100.0, value=50.0, step=1.0)
glucose = st.number_input('Average glucose level (mg/dL)', min_value=50.0, max_value=300.0, value=100.0, step=1.0)
bmi_known = st.checkbox('BMI is known', value=True)
bmi_input = st.number_input('BMI', min_value=10.0, max_value=100.0, value=28.0, step=0.1, disabled=not bmi_known)
hypertension = st.checkbox('Hypertension')
heart_disease = st.checkbox('Heart disease')
gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
ever_married = st.selectbox('Ever married', ['No', 'Yes'])
work_type = st.selectbox('Work type', ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
residence = st.selectbox('Residence type', ['Rural', 'Urban'])
smoking = st.selectbox('Smoking status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])
def get_age_group(value):
    for upper, label in zip(config['age_bins'][1:], config['age_labels']):
        if value <= upper:
            return label
if st.button('Calculate'):
    
    bmi = bmi_input if bmi_known else config['bmi_median_by_age_group'][get_age_group(age)]

    row = {
        'age': age,
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'avg_glucose_level': glucose,
        'bmi': bmi,
        'bmi_missing': not bmi_known,
        'gender_Male': gender == 'Male',
        'gender_Other': gender == 'Other',
        'ever_married_Yes': ever_married == 'Yes',
        'work_type_Never_worked': work_type == 'Never_worked',
        'work_type_Private': work_type == 'Private',
        'work_type_Self-employed': work_type == 'Self-employed',
        'work_type_children': work_type == 'children',
        'Residence_type_Urban': residence == 'Urban',
        'smoking_status_formerly smoked': smoking == 'formerly smoked',
        'smoking_status_never smoked': smoking == 'never smoked',
        'smoking_status_smokes': smoking == 'smokes',
    }
    X = pd.DataFrame([row])[config['columns']]
    score = model.predict_proba(X)[0, 1]

    st.metric('Model score', f'{score:.3f}')
    if score >= config['threshold']:
        st.error(f"Above the working threshold ({config['threshold']}): flagged as higher risk.")
    else:
        st.success(f"Below the working threshold ({config['threshold']}): not flagged.")
    st.caption('The score reflects patterns in one Kaggle dataset. At this threshold, about 5.5 patients were flagged incorrectly for each stroke found on the test set (recall 0.80, precision 0.16).')
    if not bmi_known:
        st.info('BMI is marked as unknown. In the training data, a missing BMI was associated with a higher stroke rate (19.9% versus 4.3%), and the model uses this. Treat the score for such cases with extra caution.')