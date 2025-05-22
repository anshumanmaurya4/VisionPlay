import os
import base64
import traceback
import logging
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Configure the Gemini API with the provided key
# This is your original API key
GEMINI_API_KEY = "AIzaSyC0I-4DGkZLanD8uHPfIy-5e69J8aUj2_M"
genai.configure(api_key=GEMINI_API_KEY)

# Global flag to indicate whether we're in fallback mode due to API quota limits
in_fallback_mode = False

# Store conversation history
conversation_history = []

# Function to generate a fallback response when API is unavailable
def generate_fallback_response(prompt, is_image_request=False):
    """Generate a fallback response when the API quota is exceeded."""
    logging.info("Using fallback response system")
    
    # Different responses for different types of queries
    if is_image_request:
        return (
            "I'm sorry, but I cannot analyze this image right now due to API quota limitations. \n\n"
            "The Google Gemini API quota has been exceeded. Here's what you can do:\n"
            "1. Try again later when the quota resets\n"
            "2. Consider upgrading the API plan\n"
            "3. In the meantime, please describe the image, and I'll try to respond based on your description."
        )
    
    # Text response fallbacks based on simple keyword matching
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! I'm operating in fallback mode due to API quota limitations. I can still try to help with simple questions."
    
    if any(word in prompt_lower for word in ['how are you', 'how do you feel']):
        return "I'm in fallback mode due to API quota limitations, but I'm still here to assist as best I can!"
    
    if 'weather' in prompt_lower:
        return "I'm unable to check the weather right now as I'm operating in fallback mode due to API quota limitations."
    
    if any(word in prompt_lower for word in ['thank', 'thanks']):
        return "You're welcome! Happy to help even in fallback mode."
    
    # Default fallback response
    return (
        "I'm currently operating in fallback mode because the Google Gemini API quota has been exceeded. \n\n"
        "This means my responses are limited to pre-programmed replies rather than dynamic AI-generated content. \n\n"
        "The API quota typically resets periodically, so please try again later. Or you could ask me simple questions that don't require the full AI capabilities."
    )

# Import any additional exceptions we need
from google.api_core.exceptions import NotFound

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global in_fallback_mode
    
    data = request.json
    message = data.get('message', '')
    image_data = data.get('image', None)
    
    try:
        logging.info(f"Processing chat request with message: {message[:50] if message else ''}...")
        logging.info(f"Image data present: {image_data is not None}")
        
        # Check if we're in fallback mode first before attempting API calls
        if in_fallback_mode:
            ai_response = generate_fallback_response(message, is_image_request=(image_data is not None))
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
                    ai_response = generate_fallback_response(message, is_image_request=True)
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
