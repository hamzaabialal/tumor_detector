from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from django.views.generic import TemplateView

from .models import TumorPrediction

import tensorflow as tf
import numpy as np
import cv2
import os

model_paths = [os.path.join(settings.BASE_DIR, 'myapp/brain_tumor_detector.h5')]


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_model_safely(path):
    try:
        model = tf.keras.models.load_model(path, compile=False)
        # Compile the model if it was loaded successfully
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {path}. Error: {e}")
        return None

from PIL import Image


def preprocess_image(image_path, input_shape):
    try:
        img = Image.open(image_path)
        img = img.resize(input_shape)
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        logger.error(f"Error while preprocessing image {image_path}: {e}")
        return None

models = [load_model_safely(path) for path in model_paths]
# Filter out any models that failed to load
models = [model for model in models if model is not None]

def predict_image(image_path):
    logger.debug(f"Attempting to predict with image at path: {image_path}")

    predictions = []
    for i, model in enumerate(models):
        input_shape = model.input_shape[1:3]  # Get the expected input shape
        logger.debug(f"Model {i+1} expected input shape: {input_shape}")

        img = preprocess_image(image_path, input_shape)
        if img is None:
            logger.debug(f"Image preprocessing failed for path: {image_path}")
            continue

        try:
            prediction = model.predict(img)
            predictions.append(prediction)
            logger.debug(f"Prediction from model {i+1}: {prediction}")
        except Exception as e:
            logger.error(f"Failed to predict with model {i+1}. Error: {e}")

    if predictions:
        average_prediction = np.mean(predictions, axis=0)
        return average_prediction
    else:
        logger.error("No predictions were made.")
        return None




from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
def tumor_page(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            uploaded_file = request.FILES['image']
            file_name = uploaded_file.name
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)

            try:
                # Save uploaded file
                with default_storage.open(file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                file_path = "C:/Users/Apple Computer/Documents/backend/tumor/423.png"
                result = predict_image(file_path)
                if result is not None:
                    final_result = result[0][0]
                    diagnosis = "Tumor Detected" if final_result > 0.50 else "No Tumor Detected"

                    # Save to the database
                    TumorPrediction.objects.create(
                        image_path=file_name,
                        result=final_result,
                        diagnosis=diagnosis
                    )

                    return render(request, 'tumor.html', {
                        'image_url': uploaded_file.url,
                        'diagnosis': diagnosis,
                        'final_result': final_result
                    })
                else:
                    return render(request, 'tumor.html', {
                        'error': "Prediction failed."
                    })

            except Exception as e:
                logger.error(f"Error saving file or predicting image: {e}")
                return render(request, 'tumor.html', {
                    'error': "An error occurred while processing the image."
                })
    return render(request, 'tumor.html')

class ElementsPageView(TemplateView):
    template_name = 'elements.html'

class IndexPageView(TemplateView):
    template_name = 'index.html'
