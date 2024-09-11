import os

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from django.views.generic import TemplateView
from django.shortcuts import render
from django.db import models

from djangoProject2 import settings
from myapp.models import TumorPrediction

# Paths to the models
model_paths = ['myapp/brain_tumor_detector.h5']


def process_and_predict_image(image_path):
    def load_model_safely(path):
        try:
            model = tf.keras.models.load_model(path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"Model loaded successfully from {path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {path}. Error: {e}")
            return None

    def preprocess_image(image_path, target_size):
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            print(f"Error preprocessing image at {image_path}. Error: {e}")
            return None

    def predict_image(image_path, models):
        predictions = []
        for i, model in enumerate(models):
            input_shape = model.input_shape[1:3]
            img = preprocess_image(image_path, input_shape)
            if img is None:
                continue

            try:
                prediction = model.predict(img)
                predictions.append(prediction)
                print(f"Prediction from model {i + 1}: {prediction}")
            except Exception as e:
                print(f"Failed to predict with model {i + 1}. Error: {e}")

        if predictions:
            average_prediction = np.mean(predictions, axis=0)
            return average_prediction
        else:
            print("No predictions were made.")
            return None

    def display_result(image_path, result):
        if result is not None:
            diagnosis = "Tumor Detected" if result > 0.50 else "No Tumor Detected"
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"{diagnosis}")
            plt.axis('off')
            return diagnosis

    models = [load_model_safely(path) for path in model_paths]
    models = [model for model in models if model is not None]

    result = predict_image(image_path, models)
    if result is not None:
        final_result = result[0][0]
        diagnosis = display_result(image_path, final_result)
        return final_result, diagnosis
    else:
        print("Prediction failed.")
        return None, None
from django.core.files.storage import default_storage

# Django TemplateView to handle the image upload and prediction
class TumorDetectionView(TemplateView):
    template_name = 'tumor.html'

    def post(self, request, *args, **kwargs):
        if 'image' in request.FILES:
            image = request.FILES['image']
            image_name = image.name.strip()
            if 'dicom' not in image_name.lower():
                return render(request, self.template_name, {'error': 'Please upload a valid image file.'})

            # Save the image
            image_path = os.path.join(settings.MEDIA_ROOT, image_name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Process the image and get the prediction
            result, diagnosis = process_and_predict_image(image_path)

            if result is not None:
                # Save the prediction result to the database
                tumor_prediction = TumorPrediction.objects.create(
                    image_path=image_name,
                    result=result,
                    diagnosis=diagnosis
                )
                image_url = default_storage.url(tumor_prediction.image_path)
            else:
                image_url = None

            context = {
                'result': result,
                'diagnosis': diagnosis,
                'image_url': image_url
            }
            print("images Url .......", image_url)
            return render(request, self.template_name, context)

        return render(request, self.template_name)

class ElementsPageView(TemplateView):
    template_name = 'elements.html'

class IndexPageView(TemplateView):
    template_name = 'index.html'


from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm

def signup_view(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home_page')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})
