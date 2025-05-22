# VisionPlay - Multimodal AI Experience

VisionPlay is a comprehensive AI application that combines chat, image recognition, generation, and analysis capabilities in a modern, interactive interface.

![VisionPlay](https://i.imgur.com/placeholder-screenshot.jpg)

## Features

### Intelligent Chat
- Interactive conversations with AI using Ollama models
- Support for multiple AI models including llama3.2:latest (default) and deepseek-coder:latest
- Rich markdown formatting for responses
- Voice input support
- Message reactions and interactive elements

### Image Recognition & Analysis
- Powerful image analysis using TensorFlow
- Ask questions about uploaded images with responses from Ollama (deepseek-coder model)
- Split interface with image upload and chat
- Image gallery for tracking processed images

### Image Generation
- Generate images using Stable Diffusion models
- Support for multiple SD models (1.5, 2.1, SDXL)
- Customizable parameters including dimensions, steps, and guidance
- Negative prompt support

### Modern UI
- Beautiful, vibrant interface with glassmorphism/neumorphism design
- Dark mode support with smooth transitions
- Interactive animations and 3D effects (using VanillaTilt)
- Responsive layout for various screen sizes
- Customizable settings

## Setup Instructions

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed locally (for AI models)
- TensorFlow setup for image analysis

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VisionPlay.git
   cd VisionPlay
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running on http://localhost:11434

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to http://localhost:5000

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **AI Models**: 
  - Ollama (llama3.2:latest, deepseek-coder:latest)
  - TensorFlow for image analysis
  - Stable Diffusion for image generation
- **UI**: 
  - Custom CSS with glassmorphism effects
  - Font Awesome icons
  - Animate.css for animations
  - VanillaTilt for 3D effects
  - Markdown rendering for responses

## Configuration

The application uses environment variables for configuration:
- Create a `.env` file in the root directory for sensitive configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
