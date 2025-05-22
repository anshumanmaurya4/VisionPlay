import os
import base64
import traceback
import logging
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from PIL import Image
import io
import numpy as np
import requests
import json

# Import our TensorFlow models module
import tensorflow_models as tf_models

# Import our Stable Diffusion module
import stable_diffusion as sd

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Configure the Gemini API with the provided key
# This is your original API key
GEMINI_API_KEY = "AIzaSyBe4p2LTNal45A0vPGcZkFRYo7SivCuoto"
genai.configure(api_key=GEMINI_API_KEY)

# Global flag to indicate whether we're in fallback mode due to API quota limits
in_fallback_mode = False

# Store conversation history
conversation_history = []

# Function to communicate with Ollama API
def query_ollama(prompt, model="llama3.2:latest", system_prompt=None, conversation_context=None):
    """
    Send a query to the local Ollama server and get a response.
    
    Args:
        prompt (str): The user's prompt
        model (str): The Ollama model to use, defaults to llama3.2:latest
        system_prompt (str, optional): System instructions for the model
        conversation_context (list, optional): Previous conversation messages for context
        
    Returns:
        str: The model's response text
    """
    logging.info(f"Querying Ollama with model: {model}")
    
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add conversation context if provided
        if conversation_context:
            payload["context"] = conversation_context
        
        # Send request to local Ollama server
        response = requests.post("http://localhost:11434/api/generate", 
                                headers=headers, 
                                data=json.dumps(payload),
                                timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from Ollama")
        else:
            logging.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"Error communicating with Ollama: {response.status_code}"
            
    except Exception as e:
        logging.error(f"Exception while querying Ollama: {str(e)}")
        return f"Failed to communicate with Ollama: {str(e)}"

# Function to generate a fallback response when API is unavailable
def generate_fallback_response(prompt, is_image_request=False, image_data=None):
    """Generate a fallback response using Ollama when the API quota is exceeded."""
    logging.info("Using Ollama fallback response system")
    
    # For image requests, use TensorFlow to analyze the image and then use Ollama to respond
    if is_image_request and image_data:
        try:
            logging.info("Processing image with TensorFlow models in fallback mode")
            
            # Process the base64 image
            if ',' in image_data:
                image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Perform recognition using our TensorFlow models
            results = tf_models.recognize_image(image)
            
            # Get formatted response from the module
            tf_response = tf_models.format_results_response(results)
            
            # Create a prompt for Ollama that includes the TensorFlow results
            system_instruction = """You are an AI assistant that helps users analyze images. 
You have access to the results of image recognition models.
Based on these results, provide a helpful and insightful response about the image.
IMPORTANT: Do not mention that you are a fallback system, do not mention Ollama, 
do not mention TensorFlow, and do not mention any API limitations or quota issues.
Just provide a natural, helpful response as if you are the primary AI system."""
            
            # Create a comprehensive prompt for Ollama
            ollama_prompt = (
                f"User's message: {prompt if prompt else 'Please analyze this image.'}\n\n"
                f"I've analyzed the image and found these objects:\n\n"
                f"{tf_response}\n\n"
                f"Please provide a helpful response about this image based on these recognition results."
                f"DO NOT mention that you're operating in fallback mode, DO NOT mention TensorFlow or Ollama,"
                f"and DO NOT mention API quota or limitations. Just respond naturally as if you are the primary AI."
            )
            
            # Get response from Ollama
            ollama_response = query_ollama(ollama_prompt, system_prompt=system_instruction)
            
            # Return only the Ollama response without mentioning TensorFlow analysis
            return ollama_response
            
        except Exception as e:
            logging.error(f"Error in TensorFlow-Ollama image processing: {str(e)}")
            logging.error(traceback.format_exc())
            
            # If TensorFlow analysis fails, fall back to a generic message
            system_instruction = """You are an AI assistant that helps users analyze images.
Do not mention any fallback systems, API limitations, or model names.
Just provide a natural, helpful response."""
            
            custom_prompt = """I can't see the image clearly. Could you please describe what's in the image
or provide more details about what you'd like to know about it?"""
            
            return query_ollama(custom_prompt, system_prompt=system_instruction)
    
    # For text queries, use the conversation history for context if available
    system_instruction = """You are an AI assistant that helps users with their questions.
Respond helpfully and conversationally to user queries.
IMPORTANT: Do not mention that you are a fallback system, do not mention Ollama, 
and do not mention any API limitations or quota issues.
Just provide a natural, helpful response as if you are the primary AI system."""
    
    # Create a context array for Ollama from conversation history if available
    context = None
    if len(conversation_history) > 0:
        # Some Ollama models retain context automatically, but we'll provide it to be safe
        context_prompt = "Here's our conversation so far:\n\n"
        for entry in conversation_history[-5:]:  # Last 5 exchanges for context
            if "user" in entry:
                context_prompt += f"User: {entry['user']}\n"
            if "ai" in entry:
                context_prompt += f"Assistant: {entry['ai']}\n"
        
        context_prompt += f"\nNow respond to this: {prompt}"
        return query_ollama(context_prompt, system_prompt=system_instruction)
    
    # If no conversation history, just query directly
    return query_ollama(prompt, system_prompt=system_instruction)

# List available models
try:
    logging.info("Listing available models...")
    for model in genai.list_models():
        logging.info(f"Found model: {model.name}")
        print(f"Available model: {model.name}")
except Exception as e:
    logging.error(f"Error listing models: {str(e)}")

# Initialize models
text_model = None
vision_model = None

# Model configuration - Using the latest models available
try:
    # Try various models in order of preference
    models_to_try = [
        'gemini-1.5-pro',  # This one appears to be available
        'gemini-pro',      # Fallback option
        'text-bison-001'   # Last resort
    ]
    
    for model_name in models_to_try:
        try:
            # Make sure to use the full model name if needed
            full_model_name = model_name
            if not model_name.startswith('models/'):
                full_model_name = f"models/{model_name}"
            text_model = genai.GenerativeModel(full_model_name)
            logging.info(f"Successfully initialized text model with '{full_model_name}'")
            break
        except Exception as e:
            logging.warning(f"Failed to initialize '{model_name}': {e}")
    else:
        # If we get here, none of the models worked, try to use any available model
        models = list(genai.list_models())
        if models:
            first_model = models[0].name
            logging.info(f"Falling back to first available model: {first_model}")
            text_model = genai.GenerativeModel(first_model)
        else:
            logging.error("No models available!")
            raise ValueError("No models available for use")
except Exception as e:
    logging.error(f"Error initializing text model: {str(e)}")
    text_model = None

# Vision model configuration
try:
    # Try various vision models in order of preference - using models that can handle both text and images
    vision_models_to_try = [
        'gemini-1.5-flash',       # Recommended replacement for vision tasks
        'gemini-1.5-pro',         # Alternative option
        'gemini-1.5-flash-latest' # Latest version
    ]
    
    for model_name in vision_models_to_try:
        try:
            # Make sure to use the full model name if needed
            full_model_name = model_name
            if not model_name.startswith('models/'):
                full_model_name = f"models/{model_name}"
            vision_model = genai.GenerativeModel(full_model_name)
            logging.info(f"Successfully initialized vision model with '{full_model_name}'")
            break
        except Exception as e:
            logging.warning(f"Failed to initialize '{model_name}': {e}")
    else:
        # If we get here, try to find any vision-capable model
        vision_models = [m for m in genai.list_models() if 'vision' in m.name.lower()]
        if vision_models:
            first_vision_model = vision_models[0].name
            logging.info(f"Falling back to first available vision model: {first_vision_model}")
            vision_model = genai.GenerativeModel(first_vision_model)
        else:
            logging.error("No vision models available, using text model for all requests")
            vision_model = text_model
except Exception as e:
    logging.error(f"Error initializing vision model: {str(e)}")
    vision_model = text_model

# Endpoint for Stable Diffusion image generation
@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model_version = data.get('model_version', '1.5')
        negative_prompt = data.get('negative_prompt', '')
        width = int(data.get('width', 512))
        height = int(data.get('height', 512))
        steps = int(data.get('steps', 30))
        guidance_scale = float(data.get('guidance_scale', 7.5))
        seed = data.get('seed')
        
        # Convert seed to int if provided
        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                seed = None
        
        logging.info(f"Generating image with prompt: {prompt[:50]}... using SD {model_version}")
        
        # Generate the image
        image_base64, error = sd.generate_image(
            prompt=prompt,
            model_version=model_version,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        if error:
            return jsonify({"error": error}), 500
        
        # Return the generated image
        return jsonify({
            "image": image_base64,
            "prompt": prompt,
            "model": model_version
        })
        
    except Exception as e:
        logging.error(f"Error in image generation: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Error generating image: {str(e)}"}), 500

# Endpoint to get available Stable Diffusion models
@app.route('/api/sd-models', methods=['GET'])
def get_sd_models():
    try:
        models = sd.get_available_models()
        return jsonify({"models": models})
    except Exception as e:
        logging.error(f"Error getting SD models: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

# Endpoint to handle voice input
@app.route('/api/voice-input', methods=['POST'])
def voice_input():
    try:
        # Check if file is included in request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Save the file temporarily
        temp_filename = os.path.join(os.path.dirname(__file__), "temp_audio.webm")
        audio_file.save(temp_filename)
        
        try:
            # Use OpenAI's Whisper model via API if available
            if not in_fallback_mode and GEMINI_API_KEY:
                # Attempt to use OpenAI Whisper through API
                transcript = "Voice input received"  # Placeholder for actual implementation
                
                # In a real implementation, you would use a speech-to-text service like:
                # transcript = openai_whisper_api(temp_filename)
                # or Google's Speech-to-Text API
            else:
                # Fallback message
                transcript = "Voice input detected but speech-to-text service unavailable"
                
            return jsonify({"transcript": transcript})
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
    except Exception as e:
        logging.error(f"Error processing voice input: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Error processing voice input: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def list_models():
    """Return a list of available TensorFlow models"""
    models_info = tf_models.get_available_models()
    
    return jsonify({
        "success": True,
        "available_models": models_info
    })

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Endpoint for image recognition using TensorFlow"""
    data = request.json
    image_data = data.get('image', None)
    model_name = data.get('model', None)  # Optional model selection
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Process the base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Perform recognition
        results = tf_models.recognize_image(image, model_name)
        
        # Format results
        formatted_results = [
            {
                "model": model,
                "class_id": class_id,
                "label": label,
                "confidence": round(confidence * 100, 2)
            }
            for model, class_id, label, confidence in results
        ]
        
        return jsonify({
            "success": True,
            "results": formatted_results
        })
    except Exception as e:
        logging.error(f"Error in recognition endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global in_fallback_mode
    
    data = request.json
    message = data.get('message', '')
    image_data = data.get('image', None)
    use_tf_recognition = data.get('use_tf_recognition', False)
    tf_model_name = data.get('tf_model', None)  # Optional model selection
    
    try:
        logging.info(f"Processing chat request with message: {message[:50] if message else ''}...")
        logging.info(f"Image data present: {image_data is not None}")
        
        # If TensorFlow recognition is requested and image is provided
        if use_tf_recognition and image_data:
            try:
                # Process image
                if ',' in image_data:
                    image_data_clean = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
                else:
                    image_data_clean = image_data
                image_bytes = base64.b64decode(image_data_clean)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Perform recognition using our module
                results = tf_models.recognize_image(image, tf_model_name)
                
                # Get formatted response from the module
                tf_response = tf_models.format_results_response(results)
                
                # If there's also a message, process it with Gemini and combine responses
                if message and not in_fallback_mode and text_model:
                    try:
                        # Create a comprehensive prompt for Gemini
                        gemini_prompt = (
                            f"{message}\n\n"
                            f"I've analyzed the image using TensorFlow and found these objects:\n"
                        )
                        
                        # Add the top 3 results to the prompt for context
                        top_results = sorted(
                            [(label, conf) for _, _, label, conf in results], 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:3]
                        
                        for label, conf in top_results:
                            gemini_prompt += f"- {label} ({round(conf * 100)}% confidence)\n"
                        
                        # Generate response from Gemini
                        gemini_response = text_model.generate_content(gemini_prompt)
                        
                        # Combine TensorFlow analysis with Gemini's response
                        ai_response = (
                            f"{tf_response}\n\n"
                            f"Gemini's analysis based on the recognition results:\n"
                            f"{gemini_response.text}"
                        )
                    except Exception as gemini_error:
                        logging.warning(f"Could not get Gemini response: {str(gemini_error)}")
                        ai_response = tf_response
                elif in_fallback_mode:
                    # Use Ollama with TensorFlow results
                    ai_response = generate_fallback_response(message, is_image_request=True, image_data=image_data)
                else:
                    ai_response = tf_response
            except Exception as img_error:
                logging.error(f"Error in TensorFlow image recognition: {str(img_error)}")
                logging.error(traceback.format_exc())
                return jsonify({"error": f"Error in image recognition: {str(img_error)}"}), 500
        # Proceed with regular Gemini processing if TF recognition not requested
        elif in_fallback_mode:
            ai_response = generate_fallback_response(message, is_image_request=(image_data is not None), image_data=image_data)
        elif image_data and vision_model:
            try:
                # Process image with text
                image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                logging.info("Successfully processed image, sending to vision model")
                
                # Configure generation parameters for better vision processing
                generation_config = {
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
                
                response = vision_model.generate_content(
                    [message, image],
                    generation_config=generation_config
                )
                ai_response = response.text
            except Exception as img_error:
                logging.error(f"Error processing image: {str(img_error)}")
                logging.error(traceback.format_exc())
                error_message = str(img_error)
                
                if "quota" in error_message.lower() or "rate limit" in error_message.lower() or "429" in error_message:
                    # This is a quota/rate limit error - switch to fallback mode
                    in_fallback_mode = True
                    ai_response = generate_fallback_response(message, is_image_request=True, image_data=image_data)
                else:                
                    return jsonify({"error": f"Error processing image: {str(img_error)}"}), 500
        elif not in_fallback_mode and text_model:
            # Process only text if not in fallback mode
            logging.info("Processing text-only request")
            try:
                response = text_model.generate_content(message)
                ai_response = response.text
            except Exception as text_error:
                logging.error(f"Error processing text: {str(text_error)}")
                logging.error(traceback.format_exc())
                error_message = str(text_error)
                
                if "quota" in error_message.lower() or "rate limit" in error_message.lower() or "429" in error_message:
                    # This is a quota/rate limit error - switch to fallback mode
                    in_fallback_mode = True
                    ai_response = generate_fallback_response(message, is_image_request=False)
                else:
                    return jsonify({"error": f"Error processing text: {str(text_error)}"}), 500
        else:
            # We're in fallback mode or models failed to initialize
            if text_model is None and vision_model is None:
                return jsonify({"error": "Models failed to initialize. Please check the API key and try again."}), 500
            else:
                ai_response = generate_fallback_response(message, is_image_request=False)
        
        # Add to conversation history
        conversation_history.append({"user": message, "ai": ai_response})
        logging.info("Successfully processed request and generated response")
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        logging.error(f"Unexpected error in chat endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Test the API key before starting the server
    try:
        logging.info("Testing API key...")
        if text_model:
            test_response = text_model.generate_content("Hello, please respond with 'API key is working' if you receive this message.")
            logging.info(f"Test response: {test_response.text}")
            print("API key test successful!")
            in_fallback_mode = False
        else:
            print("Cannot test API key because models failed to initialize.")
            in_fallback_mode = True
    except Exception as e:
        logging.error(f"Error testing API key: {str(e)}")
        logging.error(traceback.format_exc())
        print(f"\nERROR: API key test failed with error: {str(e)}\n")
        print("Application will run in fallback mode with limited functionality.")
        in_fallback_mode = True
    
    app.run(debug=True)
