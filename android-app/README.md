# AI Chat Android App

This is the Android wrapper for the AI Chat Multimodal application. The app uses a WebView to display and interact with the Flask-based web application.

## Setup Instructions

### Prerequisites

1. Android Studio installed on your development machine
2. Basic familiarity with Android development
3. Your Flask app running either locally or on a server

### Steps to Import and Run

1. Open Android Studio
2. Select "Open an Existing Project"
3. Navigate to this folder and open it
4. Wait for Gradle sync to complete
5. Connect an Android device or start an emulator
6. Click the "Run" button

### Configuration

- The app is configured to connect to `http://10.0.2.2:5000/` which points to localhost:5000 on your development machine when using an Android emulator.
- If you're using a physical device or deploying to production, you'll need to modify the URL in `MainActivity.java` to point to your hosted Flask application.

## Deployment

To create a production APK:

1. In Android Studio, select Build > Build Bundle(s) / APK(s) > Build APK(s)
2. The APK will be generated and saved in the `app/build/outputs/apk/debug/` directory
3. This APK can be installed on Android devices (Android 5.0 and above)

## Hosting Your Flask App

For a production version of the app, you'll need to host your Flask application on a server with a publicly accessible URL. Some options include:

- Heroku
- Google Cloud Platform
- AWS
- Digital Ocean
- PythonAnywhere (specifically good for Flask/Django apps)

Remember to update the URL in `MainActivity.java` to point to your hosted application.

## Permissions

The app requires the following permissions:

- Internet access
- Camera access
- Storage access (for image uploads)

Users will be prompted to grant these permissions when they first launch the app.
