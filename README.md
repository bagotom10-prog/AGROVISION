# 🌾 AgroVision AI: Real-Time Plant Pathology Detection

AgroVision AI es una aplicación web diseñada para la detección en tiempo real de enfermedades y plagas en cultivos agrícolas mediante visión por computador y Deep Learning. Utiliza un modelo personalizado basado en **YOLOv8** para identificar patologías en hojas a partir de fotografías tomadas directamente en el campo.

---

## 🚀 Características Principales

* **Detección Automática:** Clasificación y localización de patologías foliares en segundos.
* **Interfaz Móvil y Web:** Accesible desde cualquier navegador web o smartphone gracias a Streamlit.
* **Modelo Ligero:** Optimizado para ofrecer inferencias rápidas y precisión en la detección.

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.10+
* **Deep Learning:** [Ultralytics YOLOv8](https://docs.ultralytics.com/)
* **Procesamiento de Imagen:** OpenCV, Pillow, NumPy
* **Interfaz Web:** [Streamlit](https://streamlit.io/)
* **Control de Versiones:** Git & GitHub

---

## 📁 Estructura del Repositorio

```text
AGROVISION/
├── runs/                   # Pesos y métricas del entrenamiento de YOLOv8
├── app.py                  # Código principal de la interfaz web Streamlit
├── export_model.py         # Script para exportar el modelo a formatos como ONNX/TFLite
├── predict.py              # Script para pruebas de inferencia locales
├── train.py                # Script de entrenamiento del modelo YOLOv8
├── requirements.txt        # Dependencias del proyecto para producción
├── .gitignore              # Archivos y carpetas excluidos del control de versiones
└── README.md               # Documentación del proyecto
