const codeSections = {
    full: {
      code: `import speech_recognition as sr
  from gtts import gTTS
  import os
  import serial
  import time
  
  # Set up serial communication with Arduino
  arduino = serial.Serial(port='COM12', baudrate=9600, timeout=1)  # Adjust 'COM12' to your Arduino's port
  
  # Initialize the recognizer
  recognizer = sr.Recognizer()
  
  def listen_command():
      with sr.Microphone() as source:
          print("Adjusting for ambient noise, please wait...")
          recognizer.adjust_for_ambient_noise(source, duration=1)
          print("Listening for command...")
          audio = recognizer.listen(source)
          try:
              command = recognizer.recognize_google(audio)
              print(f"Command received: {command}")
              return command.upper()
          except sr.UnknownValueError:
              print("Sorry, I did not understand that.")
              return None
          except sr.RequestError:
              print("Could not request results; check your network connection.")
              return None
  
  def execute_command(command):
      try:
          angle = int(command.split()[-1])
          angle = round(angle / 5) * 5  # Round to nearest multiple of 5
          if 0 <= angle <= 180:
              arduino.write(f"{angle}\\n".encode())
              speak(f"Turning to {angle} degrees")
          else:
              speak("Angle must be between 0 and 180 degrees")
      except ValueError:
          speak("Please specify a valid angle")
  
  def speak(text):
      tts = gTTS(text=text, lang='en')
      tts.save("response.mp3")
      os.system("start response.mp3")  # For Windows, use 'start response.mp3'; for Mac use 'afplay response.mp3'
  
  while True:
      command = listen_command()
      if command:
          execute_command(command)
      time.sleep(2)  # Short delay to avoid multiple rapid commands
  `,
      language: "python"
    },
    arduino_code: {
      code: `#include <Servo.h>
  
  Servo myServo;
  const int servoPin = 9;
  
  void setup() {
    myServo.attach(servoPin);
    Serial.begin(9600); // Communication with the computer
  }
  
  void loop() {
    if (Serial.available() > 0) {
      String command = Serial.readStringUntil('\\n');
      int angle = command.toInt();
      if (angle >= 0 && angle <= 180) {
        myServo.write(angle); // Set the servo to the specified angle
      }
    }
  }
  `,
      language: "c_cpp"
    },
    python_code: {
      code: `import speech_recognition as sr
  from gtts import gTTS
  import os
  import serial
  import time
  
  # Set up serial communication with Arduino
  arduino = serial.Serial(port='COM12', baudrate=9600, timeout=1)  # Adjust 'COM12' to your Arduino's port
  
  # Initialize the recognizer
  recognizer = sr.Recognizer()
  
  def listen_command():
      with sr.Microphone() as source:
          print("Adjusting for ambient noise, please wait...")
          recognizer.adjust_for_ambient_noise(source, duration=1)
          print("Listening for command...")
          audio = recognizer.listen(source)
          try:
              command = recognizer.recognize_google(audio)
              print(f"Command received: {command}")
              return command.upper()
          except sr.UnknownValueError:
              print("Sorry, I did not understand that.")
              return None
          except sr.RequestError:
              print("Could not request results; check your network connection.")
              return None
  
  def execute_command(command):
      try:
          angle = int(command.split()[-1])
          angle = round(angle / 5) * 5  # Round to nearest multiple of 5
          if 0 <= angle <= 180:
              arduino.write(f"{angle}\\n".encode())
              speak(f"Turning to {angle} degrees")
          else:
              speak("Angle must be between 0 and 180 degrees")
      except ValueError:
          speak("Please specify a valid angle")
  
  def speak(text):
      tts = gTTS(text=text, lang='en')
      tts.save("response.mp3")
      os.system("start response.mp3")  # For Windows, use 'start response.mp3'; for Mac use 'afplay response.mp3'
  
  while True:
      command = listen_command()
      if command:
          execute_command(command)
      time.sleep(2)  # Short delay to avoid multiple rapid commands
  `,
      language: "python"
    }
  };
  
  export default codeSections;
