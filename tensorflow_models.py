import os
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tensorflow.keras.applications import (
    EfficientNetB7, 
    InceptionResNetV2, 
    ResNet152V2,
    DenseNet201,
    NASNetLarge
)
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Configure logging
logger = logging.getLogger('tensorflow_models')

# Initialize TensorFlow image recognition models
tf_models = {}
model_preprocessors = {}

def load_tf_models():
    """Load multiple TensorFlow models for image recognition"""
    global tf_models, model_preprocessors
    
    models_to_load = {
        'efficientnet': {
            'class': EfficientNetB7,
            'preprocess': efficientnet_preprocess,
            'input_size': (600, 600),
            'description': 'EfficientNetB7 - Very high accuracy with relatively efficient computation'
        },
        'inception': {
            'class': InceptionResNetV2,
            'preprocess': inception_preprocess,
            'input_size': (299, 299),
            'description': 'InceptionResNetV2 - High accuracy hybrid model combining Inception and ResNet architectures'
        },
        'resnet': {
            'class': ResNet152V2,
            'preprocess': resnet_preprocess,
            'input_size': (224, 224),
            'description': 'ResNet152V2 - Deep residual network with 152 layers'
        },
        'densenet': {
            'class': DenseNet201, 
            'preprocess': densenet_preprocess,
            'input_size': (224, 224),
            'description': 'DenseNet201 - Dense connections between layers, good feature propagation'
        }
    }
    
    # Try to load all models, but ensure at least one is loaded
    for model_name, model_config in models_to_load.items():
        try:
            logger.info(f"Loading {model_name} model...")
            model = model_config['class'](weights='imagenet', include_top=True)
            tf_models[model_name] = model
            model_preprocessors[model_name] = {
                'preprocess': model_config['preprocess'],
                'input_size': model_config['input_size'],
                'description': model_config['description']
            }
            logger.info(f"Successfully loaded {model_name} model")
        except Exception as e:
            logger.error(f"Error loading {model_name} model: {str(e)}")
    
    if not tf_models:
        logger.error("Failed to load any TensorFlow models!")
        # As a fallback, try to load a smaller model
        try:
            tf_models['mobilenet'] = MobileNetV2(weights='imagenet')
            model_preprocessors['mobilenet'] = {
                'preprocess': preprocess_input,
                'input_size': (224, 224),
                'description': 'MobileNetV2 - Lightweight model (fallback)'
            }
            logger.info("Loaded MobileNetV2 as fallback model")
        except Exception as e:
            logger.error(f"Failed to load fallback model: {str(e)}")
    
    return len(tf_models) > 0

def get_available_models():
    """Return information about available models"""
    models_info = {}
    for name, config in model_preprocessors.items():
        models_info[name] = {
            "description": config['description'],
            "input_size": f"{config['input_size'][0]}x{config['input_size'][1]}"
        }
    return models_info

def enhance_image(img):
    """Apply image enhancement techniques to improve recognition quality"""
    try:
        # Convert PIL image to OpenCV format
        img_cv = np.array(img)
        img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
        
        # Apply enhancements
        # 1. Automatic brightness and contrast optimization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Convert back to RGB and then to PIL
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_img)
    except Exception as e:
        logger.warning(f"Image enhancement failed: {str(e)}. Using original image.")
        return img

def recognize_image(image_input, model_name=None):
    """
    Perform image recognition using specified TensorFlow model.
    If model_name is None, use all available models and ensemble the results.
    Returns a list of (model_name, class_id, description, probability) tuples.
    """
    if not tf_models:
        return [("Error", "Error", "No TensorFlow models available", 1.0)]
    
    try:
        # Apply image enhancement
        enhanced_img = enhance_image(image_input)
        
        # If no specific model is requested, use ensemble approach
        if model_name is None or model_name not in tf_models:
            # Aggregate results from all models
            all_results = []
            
            for name, model in tf_models.items():
                try:
                    config = model_preprocessors[name]
                    # Resize image as required by the model
                    img = enhanced_img.resize(config['input_size'])
                    # Convert to array and add batch dimension
                    img_array = np.array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    # Preprocess the image
                    img_array = config['preprocess'](img_array)
                    
                    # Get predictions
                    predictions = model.predict(img_array)
                    # Decode top predictions
                    results = decode_predictions(predictions, top=5)[0]
                    model_results = [(name, class_id, label, float(score)) for class_id, label, score in results]
                    all_results.extend(model_results)
                except Exception as e:
                    logger.error(f"Error with model {name}: {str(e)}")
                    continue
            
            if not all_results:
                return [("Error", "Error", "All models failed to process the image", 1.0)]
            
            # Sort by confidence and take top 10 unique results
            all_results.sort(key=lambda x: x[3], reverse=True)
            
            # Remove duplicates (same label from different models, keep highest confidence)
            unique_results = {}
            for model, class_id, label, score in all_results:
                if label not in unique_results or score > unique_results[label][3]:
                    unique_results[label] = (model, class_id, label, score)
            
            # Convert back to list and take top 10
            final_results = [(model, class_id, label, score) for model, class_id, label, score in unique_results.values()]
            final_results.sort(key=lambda x: x[3], reverse=True)
            return final_results[:10]
        else:
            # Use specific model
            model = tf_models[model_name]
            config = model_preprocessors[model_name]
            
            # Resize image as required by the model
            img = enhanced_img.resize(config['input_size'])
            # Convert to array and add batch dimension
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            # Preprocess the image
            img_array = config['preprocess'](img_array)
            
            # Get predictions
            predictions = model.predict(img_array)
            # Decode top 10 predictions
            results = decode_predictions(predictions, top=10)[0]
            return [(model_name, class_id, label, float(score)) for class_id, label, score in results]
    except Exception as e:
        logger.error(f"Error during image recognition: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [("Error", "Error", f"Recognition failed: {str(e)}", 1.0)]

# Function to format recognition results into a readable response
def format_results_response(results):
    """Format recognition results into a readable text response"""
    if not results or results[0][0] == "Error":
        return "Sorry, I couldn't recognize anything in this image."
    
    # Group results by model
    results_by_model = {}
    for model, class_id, label, confidence in results:
        if model not in results_by_model:
            results_by_model[model] = []
        results_by_model[model].append((label, confidence))
    
    # Format the response
    response = "I've analyzed this image using advanced TensorFlow models and here's what I found:\n\n"
    
    # Format results by model
    for model, model_results in results_by_model.items():
        model_description = model_preprocessors.get(model, {}).get('description', model)
        response += f"Model: {model_description}\n"
        for label, confidence in model_results:
            percentage = round(confidence * 100, 2)
            response += f"- {label}: {percentage}% confidence\n"
        response += "\n"
    
    return response

# Initialize models when module is imported
if __name__ != "__main__":
    # Only initialize when imported as a module
    success = load_tf_models()
    if success:
        logger.info("TensorFlow models loaded successfully")
    else:
        logger.error("Failed to load TensorFlow models")
