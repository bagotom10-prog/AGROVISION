from ultralytics import YOLO

# 1. Cargamos el modelo que acabas de entrenar (tus pesos personalizados)
model = YOLO('runs/detect/PlantDoc_yolo_run/weights/best.pt')

# 2. Elegimos una imagen de prueba del dataset (puedes cambiar la ruta por cualquier otra foto de tu carpeta test)
# Vamos a buscar una imagen dentro de la carpeta 'PlantDoc/test/images'
import os
test_images_dir = 'PlantDoc/test/images'
image_files = os.listdir(test_images_dir)

if len(image_files) > 0:
    image_path = os.path.join(test_images_dir, image_files[7])
    print(f"Probando el modelo con la imagen: {image_path}")

    # 3. Realizamos la predicción
    # save=True guarda la imagen resultante en una carpeta de runs/predict
    results = model.predict(source=image_path, save=True, conf=0.10)

    print("¡Predicción completada con éxito!")
    print("Revisa la carpeta 'runs/predict' en tu explorador para ver la imagen con las cajitas dibujadas.")
else:
    print("No se han encontrado imágenes en la carpeta de test.")