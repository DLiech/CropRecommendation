import pickle
import numpy as np
import streamlit as st
import os

def calculate_ratios(N, P, K):
    ratio_N_P = N / P if P != 0 else 0
    ratio_P_K = P / K if K != 0 else 0
    ratio_K_N = K / N if N != 0 else 0
    return ratio_N_P, ratio_P_K, ratio_K_N

def getCropDict():
    crop_dict = { 
        'rice': 1,
        'maize': 2,
        'jute': 3,
        'cotton': 4,
        'coconut': 5,
        'papaya': 6,
        'orange': 7,
        'apple': 8,
        'muskmelon': 9,
        'watermelon': 10,
        'grapes': 11,
        'mango': 12,
        'banana': 13,
        'pomegranate': 14,
        'lentil': 15,
        'blackgram': 16,
        'mungbean': 17,
        'mothbeans': 18,
        'pigeonpeas': 19,
        'kidneybeans': 20,
        'chickpea': 21,
        'coffee': 22
    }

    # Create a reverse dictionary
    reverse_crop_dict = {value: key for key, value in crop_dict.items()}
    return reverse_crop_dict

def getUserInput():
    directory = 'Model'

    model_collection = []   
    for model in  os.listdir(directory):
        if os.path.isfile(os.path.join(directory, model)):
            scorePercentage = float(model.split('_')[0]) * 100
            model_name = model.split('_')[1]

            formatted_string = f"{model_name} || Accuracy: {scorePercentage:.4f}%"
            model_collection.append(formatted_string)

    modelName = st.selectbox("Choose our model:", model_collection, format_func=lambda x: x, placeholder="Select a model")

    N = st.number_input("Ratio of Nitrogen in soil", min_value=0, max_value=100, value=1)
    P = st.number_input("Ratio of Phosphorous in soil", min_value=0, max_value=100, value=1)
    K = st.number_input("Ratio of Potassium in soil", min_value=0, max_value=100, value=1)
    temperature = st.number_input("Average temperature in Celsius", value=23.0, format="%.2f")
    humidity = st.number_input("Relative humidity in %", min_value=0, max_value=100, value=50)
    ph = st.number_input("pH of soil", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall in mm", value=100.0, format="%.2f")

    return modelName, N, P, K, temperature, humidity, ph, rainfall

def process(modelName, N, P, K, temperature, humidity, ph, rainfall):
    ratio_N_P, ratio_P_K, ratio_K_N = calculate_ratios(N, P, K)
    input_features = [N, P, K, temperature, humidity, ph, rainfall, ratio_N_P, ratio_P_K, ratio_K_N]
    input_array = np.array([input_features])
    model_name = modelName.split(' || Accuracy: ')[0]
    score = modelName.split(' || Accuracy: ')[1]
    score_percentage = float(score.replace('%', '')) / 100

    model_path = f'Model/{score_percentage:.4f}_{model_name}_model.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    crop_dict = getCropDict()
    predicted_crop_label = model.predict(input_array)
    predicted_crop_name = crop_dict[predicted_crop_label[0]]

    st.markdown(f"<div style='text-align: center; font-weight: bold;'>The recommended crop: </div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; font-weight: bold;'> {predicted_crop_name.title()} </div>", unsafe_allow_html=True)

def css():
    st.markdown("""
        <style>
        .stButton {
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    css()

    st.title("Crop Recommendation System")

    modelName, N, P, K, temperature, humidity, ph, rainfall = getUserInput()
    
    with st.container():
        if st.button("Predict Now"):
            process(modelName, N, P, K, temperature, humidity, ph, rainfall)

main()
