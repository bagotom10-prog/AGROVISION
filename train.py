from ultralytics import YOLO

print("Cargando modelo base de YOLOv8...")
# Cargamos la versión nano ('n') para que el entrenamiento sea ágil en local. Si usamos una version mayor aprendera mejor pero tardara mucho mas tiempo y el pc se calentara demasiado.
model = YOLO('yolov8n.pt') 

print("Iniciando el entrenamiento con el dataset PlantDoc...")
# Entrenamos apuntando al archivo de configuración yaml de tu carpeta
results = model.train(
    data='PlantDoc/data.yaml',  # Ruta hacia el archivo data.yaml dentro de plantdoc
    epochs=10,                  # Empezamos con 10 épocas para probar, pero son pocas, le deberiamos dar mas (around 50) para mejorar los resultados. No lo he hecho porque tardaria horas y ademas el pc se calienta mucho
    imgsz=416,                  # Tamaño de imagen preprocesado por Roboflow
    batch=16,                   # Lote de procesamiento
    name='PlantDoc_yolo_run'    # Nombre de la carpeta donde se guardarán los resultados
)

print("¡Entrenamiento finalizado con éxito!")