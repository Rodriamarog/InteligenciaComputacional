# Proyecto Red Neuronal MNIST

Este proyecto implementa una red neuronal feedforward para clasificar dígitos del dataset MNIST usando PyTorch.

## Configuración del Entorno

1. Activar el entorno virtual:
```bash
source mnist_env/bin/activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Abrir Jupyter Notebook:
```bash
jupyter notebook
```

## Estructura del Proyecto

- `1.mnist_neural_network.ipynb`: Notebook principal con toda la implementación
- `requirements.txt`: Dependencias del proyecto
- `mnist_env/`: Entorno virtual
- `data/`: Dataset MNIST (se crea automáticamente)

## Objetivo

Implementar y entrenar al menos 3 redes neuronales diferentes variando:
- Arquitectura (número de capas y neuronas)
- Optimizador (SGD, Adam)
- Hiperparámetros (learning rate, batch size, épocas)

Evaluar el mejor modelo en el conjunto de test y generar reportes de clasificación.

## Restricciones

- Solo se permite el uso de PyTorch como librería de ML
- Se permiten librerías auxiliares para visualización: matplotlib, tqdm
- No se puede usar sklearn ni otras librerías de ML
- Toda funcionalidad debe implementarse con PyTorch