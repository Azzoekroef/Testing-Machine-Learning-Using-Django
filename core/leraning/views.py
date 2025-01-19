from django.shortcuts import render
from rest_framework import status
import tensorflow as tf
from rest_framework.response import Response
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import AllowAny
from django.conf import settings
import numpy as np
import dill
from tensorflow import keras

import joblib
import os
# Create your views here.

tokenizer_path = os.path.join(os.path.join('tokenizer.pkl'))
with open(tokenizer_path, 'rb') as f:
    tokenizer = dill.load(f)
# tokenizer = joblib.load(tokenizer_path)


def preprocess_text(text):
    # Convert the text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences to ensure consistent input size (adjust maxlen as needed)
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=87)
    return padded_sequences

@api_view(['POST'])
@permission_classes([AllowAny,])
def predict(request):
    text = request.data['text']
    print(os.getcwd())
    model_path = os.path.join(os.path.join('model (2).h5'))
    model = keras.models.load_model(model_path)
    input_data = preprocess_text(text)
    predictions = model.predict(input_data)
    array1 = predictions[0][0]  # First array
    array2 = predictions[1][0]  # Second array
    print(array1)
    print(array2)
    ar1 = ""
    ar2 = ""
    print(tf.__version__)
    max_index_label = np.argmax(array1)
    if max_index_label == 0:
        ar1 = "Food"
    elif max_index_label == 1:
        ar1 = "Service"
    elif max_index_label == 2:
        ar1 = "Ambiance"
    else:
        ar1 = "Price"
    max_index = np.argmax(array2)
    if max_index == 2:
        ar2 = "positif"
    elif max_index == 1:
        ar2 = "negatif"
    else:
        ar2 = "netral"
    response = {
        'labels': ar1,
        'rating': ar2
    }
    
    return Response(response)

@api_view(['GET'])
@permission_classes([AllowAny,])
def index(request):
    return render(request, 'index.html')