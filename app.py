import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuración de la página web
st.set_page_config(page_title="AgroVision AI - Detección de Plagas", page_icon="🌱", layout="centered")

st.title("🌱 AgroVision AI: Detector de Enfermedades en Cultivos")
st.write("Sube una foto de una hoja tomada en el campo para que la Inteligencia Artificial detecte posibles patologías en tiempo real.")

# Cargar el modelo entrenado (cacheado para que no se recargue en cada clic)
@st.cache_resource
def load_model():
    return YOLO('runs/detect/PlantDoc_yolo_run/weights/best.pt')

model = load_model()

# Subir archivo de imagen
uploaded_file = st.file_uploader("Elige una imagen de una hoja...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen subida por el usuario
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original cargada", use_container_width=True)
    
    if st.button("Ejecutar Diagnóstico"):
        with st.spinner("Analizando la hoja con el modelo de YOLOv8..."):
            # Ejecutar predicción
            results = model.predict(source=np.array(image), conf=0.25)
            
            # Obtener la imagen con las cajitas dibujadas
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1]) # Convertir BGR a RGB
            
            st.success("¡Análisis completado!")
            st.image(res_image, caption="Resultado de la Detección", use_container_width=True)