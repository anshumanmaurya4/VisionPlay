# Setting Up the AI Chat Android App

This document provides detailed instructions on how to import, build, and run the Android app.

## Prerequisites

1. Install Android Studio: Download and install [Android Studio](https://developer.android.com/studio) if you haven't already.
2. Basic Android knowledge: Familiarity with Android development is helpful.
3. Running Flask server: Make sure your Flask application is running either locally or on a server.

## Step 1: Import the Project

1. Open Android Studio
2. Select "Open an Existing Project" from the welcome screen
3. Navigate to the `android-app` folder and select it
4. Wait for the project to load and Gradle to sync

## Step 2: Configure the Project

1. **Update Server URL:**
   - Open `MainActivity.java`
   - Find the line: `webView.loadUrl("http://10.0.2.2:5000/");`
   - Replace with your Flask app URL:
     - For local testing with emulator, keep it as `10.0.2.2:5000`
     - For physical devices on same network: use your computer's local IP, e.g., `192.168.1.x:5000`
     - For production: use your hosted server URL, e.g., `https://your-app-name.herokuapp.com`

2. **Generate App Icon (Optional):**
   - Right-click on the `res` folder
   - Select "New > Image Asset"
   - Follow the wizard to create a custom icon

## Step 3: Build and Run

1. **Connect a device or start an emulator:**
   - Connect an Android device via USB with debugging enabled, or
   - Start an Android emulator from the AVD Manager

2. **Run the app:**
   - Click the green "Run" button in the toolbar
   - Select your device/emulator
   - Wait for the app to build and install

## Step 4: Create APK for Distribution

To generate an APK that can be shared with others:

1. Go to Build > Build Bundle(s) / APK(s) > Build APK(s)
2. Once built, click on the notification to locate the APK
3. The APK will be in `app/build/outputs/apk/debug/app-debug.apk`
4. This file can be shared and installed on Android devices

## Troubleshooting

- **App not loading content**: Make sure your Flask app is running and accessible from the device
- **Permission issues**: Ensure all required permissions are granted in device settings
- **Network errors**: Check if the device has internet access and can reach the server
- **Build errors**: Make sure Android Studio is up to date and Gradle sync completes successfully

## Next Steps

- Consider implementing push notifications
- Add offline capabilities
- Implement a native UI for critical features
- Publish to Google Play Store
