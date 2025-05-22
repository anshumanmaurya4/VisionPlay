import os
import torch
import logging
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Global variables to store loaded models
sd_models = {}

def load_stable_diffusion_model(model_version="1.5"):
    """
    Load a Stable Diffusion model
    
    Args:
        model_version (str): Version of Stable Diffusion to use ("1.5", "2.1" or "sdxl")
        
    Returns:
        The loaded pipeline or None if loading failed
    """
    global sd_models
    
    logging.info(f"Loading Stable Diffusion {model_version} model")
    
    # Check if the model is already loaded
    if model_version in sd_models and sd_models[model_version] is not None:
        logging.info(f"Using already loaded SD {model_version} model")
        return sd_models[model_version]
    
    try:
        # Select the appropriate model based on the version
        if model_version == "1.5":
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        elif model_version == "2.1":
            model_id = "stabilityai/stable-diffusion-2-1"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        elif model_version == "sdxl":
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        else:
            logging.error(f"Unsupported model version: {model_version}")
            return None

        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            logging.info("Model loaded on GPU")
        else:
            logging.info("CUDA not available, using CPU (this will be slow)")
        
        # Enable attention slicing for lower memory usage if needed
        pipe.enable_attention_slicing()
        
        # Store the loaded model for future use
        sd_models[model_version] = pipe
        
        return pipe
    
    except Exception as e:
        logging.error(f"Failed to load Stable Diffusion model: {str(e)}")
        return None

def unload_model(model_version=None):
    """
    Unload a model to free up memory
    
    Args:
        model_version (str): Version to unload, if None unload all models
    """
    global sd_models
    
    if model_version is None:
        # Unload all models
        for version in list(sd_models.keys()):
            if sd_models[version] is not None:
                sd_models[version] = None
                torch.cuda.empty_cache()
                logging.info(f"Unloaded SD {version} model")
        sd_models = {}
    elif model_version in sd_models:
        # Unload specific model
        sd_models[model_version] = None
        torch.cuda.empty_cache()
        logging.info(f"Unloaded SD {model_version} model")

def generate_image(prompt, model_version="1.5", negative_prompt="", width=512, height=512, 
                  num_inference_steps=30, guidance_scale=7.5, seed=None):
    """
    Generate an image using Stable Diffusion
    
    Args:
        prompt (str): Text prompt for image generation
        model_version (str): Version of SD to use ("1.5", "2.1", or "sdxl")
        negative_prompt (str): Negative prompt to guide what not to generate
        width (int): Image width (must be multiple of 8)
        height (int): Image height (must be multiple of 8)
        num_inference_steps (int): Number of diffusion steps
        guidance_scale (float): How closely to follow the prompt
        seed (int): Random seed for reproducibility
        
    Returns:
        Tuple of (image as base64 string, error message if any)
    """
    try:
        # Load the model
        pipe = load_stable_diffusion_model(model_version)
        if pipe is None:
            return None, "Failed to load Stable Diffusion model"
        
        # Ensure width and height are valid
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # Set the seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        else:
            generator = None
            
        # Generate the image
        if model_version == "sdxl":
            # SDXL requires a different approach
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        else:
            # Standard approach for SD 1.5 and 2.1
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # Convert the image to base64 for web display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}", None
        
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        return None, f"Error generating image: {str(e)}"

def get_available_models():
    """Return information about available Stable Diffusion models"""
    models_info = [
        {
            "id": "1.5",
            "name": "Stable Diffusion 1.5",
            "description": "The original stable diffusion model with good overall quality",
            "resolution": "512x512",
            "loaded": "1.5" in sd_models and sd_models["1.5"] is not None
        },
        {
            "id": "2.1",
            "name": "Stable Diffusion 2.1",
            "description": "Improved model with better image quality and prompt following",
            "resolution": "768x768",
            "loaded": "2.1" in sd_models and sd_models["2.1"] is not None
        },
        {
            "id": "sdxl",
            "name": "Stable Diffusion XL",
            "description": "The latest model with significantly improved image quality and detail",
            "resolution": "1024x1024",
            "loaded": "sdxl" in sd_models and sd_models["sdxl"] is not None
        }
    ]
    return models_info
