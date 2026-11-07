from ultralytics import YOLO

print("Cargando tu modelo entrenado...")
# Cargamos tus pesos personalizados
model = YOLO('runs/detect/PlantDoc_yolo_run/weights/best.pt')

print("Exportando el modelo a formato TensorFlow Lite para móviles...")
# Exportamos a tflite (puedes cambiar a 'onnx' si prefieres otro estándar)
success = model.export(format='onnx')

print(f"¡Exportación completada con éxito!: {success}")