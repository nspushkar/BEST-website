const codeSections = {
    full: {
      code: `import speech_recognition as sr
  from gtts import gTTS
  import os
  import serial
  import time
  
  const int ledPin = 7;  // Pin to which the LED is connected
  
  void setup() {
    pinMode(ledPin, OUTPUT);  // Set the LED pin as an output
    digitalWrite(ledPin, LOW);  // Ensure the LED is off at startup
    Serial.begin(9600);  // Start serial communication at 9600 baud
  }
  
  void loop() {
    if (Serial.available() > 0) {
      String command = Serial.readStringUntil('\\n');  // Read the command from serial input
      command.trim();  // Remove any leading or trailing whitespace
      if (command == "SWITCH ON") {
        digitalWrite(ledPin, HIGH);  // Switch on the LED
      } else if (command == "SWITCH OFF") {
        digitalWrite(ledPin, LOW);  // Switch off the LED
      }
    }
  }
  `,
      language: 'python',
    },
    arduino_setup: {
      code: `import serial
  import time
  
  # Set up serial communication with Arduino
  arduino = serial.Serial(port='COM12', baudrate=9600, timeout=1)  # Adjust 'COM12' to your Arduino's port
  `,
      language: 'python',
    },
    connect_led: {
      code: `const int ledPin = 7;  // Pin to which the LED is connected
  
  void setup() {
    pinMode(ledPin, OUTPUT);  // Set the LED pin as an output
    digitalWrite(ledPin, LOW);  // Ensure the LED is off at startup
    Serial.begin(9600);  // Start serial communication at 9600 baud
  }
  `,
      language: 'arduino',
    },
    microphone: {
      code: `import speech_recognition as sr
  from gtts import gTTS
  import os
  
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
  `,
      language: 'python',
    },
    arduino_code: {
      code: `void loop() {
    if (Serial.available() > 0) {
      String command = Serial.readStringUntil('\\n');  // Read the command from serial input
      command.trim();  // Remove any leading or trailing whitespace
      if (command == "SWITCH ON") {
        digitalWrite(ledPin, HIGH);  // Switch on the LED
      } else if (command == "SWITCH OFF") {
        digitalWrite(ledPin, LOW);  // Switch off the LED
      }
    }
  }
  `,
      language: 'arduino',
    },
    python_script: {
      code: `def execute_command(command):
      if command == "SWITCH ON":
          arduino.write(b"SWITCH ON\\n")
          speak("Switching on the light")
      elif command == "SWITCH OFF":
          arduino.write(b"SWITCH OFF\\n")
          speak("Switching off the light")
      else:
          speak("Sorry, I didn't understand the command.")
  
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
      language: 'python',
    },
    math_backing: {
      code: `# No code snippet required for this section as it explains the theory behind speech recognition
  `,
      language: 'text',
    },
  };
  
  export default codeSections;
  