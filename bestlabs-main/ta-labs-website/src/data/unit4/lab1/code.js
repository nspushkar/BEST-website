const codeSections = {
    full: {
      code: `from flask import Flask, request, render_template_string
  
  app = Flask(__name__)
  
  class BajajExpertSystem:
      def __init__(self):
          self.rules = [
              {"conditions": ["not_starting", "battery_low"], "conclusion": "Charge the battery."},
              {"conditions": ["not_starting", "battery_ok"], "conclusion": "Check the starter motor."},
              {"conditions": ["starting", "stalls_frequently"], "conclusion": "Check the fuel supply."},
              {"conditions": ["starting", "poor_acceleration"], "conclusion": "Check the air filter."},
              {"conditions": ["starting", "unusual_noises"], "conclusion": "Check the engine."},
              {"conditions": ["starting", "engine_overheating"], "conclusion": "Check the coolant level and radiator condition."},
              {"conditions": ["not_starting", "smoke_exhaust"], "conclusion": "Check for oil leakage or exhaust system blockages."},
              {"conditions": ["starting", "vibrating_excessively"], "conclusion": "Inspect engine mountings and wheel alignment."},
              {"conditions": ["starting", "battery_ok", "lights_dim"], "conclusion": "Check the alternator and electrical connections."},
              {"conditions": ["not_starting", "clicking_sound"], "conclusion": "Replace the starter solenoid."},
              {"conditions": ["starting", "leaking_fluids"], "conclusion": "Identify the fluid type and source of the leak."},
              {"conditions": ["starting", "brake_issues"], "conclusion": "Inspect brake pads, discs, and fluid levels."}
          ]
  
      def diagnose(self, symptoms):
          for rule in self.rules:
              if all(condition in symptoms for condition in rule["conditions"]):
                  return rule["conclusion"]
          return "No diagnosis found. Please consult a professional mechanic."
  
  @app.route('/', methods=['GET', 'POST'])
  def home():
      system = BajajExpertSystem()
      if request.method == 'POST':
          symptoms = request.form.getlist('symptoms')
          diagnosis = system.diagnose(symptoms)
          return render_template_string("""
              <!DOCTYPE html>
              <html>
              <head>
                  <style>
                      body {
                          font-family: 'Arial', sans-serif;
                          background-color: #000;
                          color: #ffa500;
                          text-align: center;
                          padding: 50px;
                      }
                      h1, p {
                          color: #ffa500;
                      }
                      .button {
                          background-color: #ff8c00;
                          color: white;
                          padding: 15px 25px;
                          margin: 10px 0;
                          border: none;
                          border-radius: 25px;
                          cursor: pointer;
                          box-shadow: 0 4px 8px 0 rgba(255,140,0,0.5);
                          transition: all 0.3s ease 0s;
                      }
                      .button:hover {
                          background-color: #ffa500;
                          box-shadow: 0 8px 16px 0 rgba(255,165,0,0.8);
                      }
                      form {
                          background-color: rgba(255, 255, 255, 0.1);
                          padding: 20px;
                          border-radius: 8px;
                          box-shadow: 0 0 10px 0 rgba(255,165,0,0.6);
                          text-align: left;
                          border: 2px solid #ff8c00; /* Dim orange border */
                          margin: auto;
                          width: fit-content;
                      }
                      label {
                          margin-right: 10px;
                          display: block;
                          color: #fff;
                      }
                  </style>
              </head>
              <body>
                  <h1>Diagnosis Result</h1>
                  <p>{{ diagnosis }}</p>
                  <a href="/" class="button">Back</a>
              </body>
              </html>
          """, diagnosis=diagnosis)
      return render_template_string("""
          <!DOCTYPE html>
          <html>
          <head>
              <style>
                  body {
                      font-family: 'Arial', sans-serif;
                      background-color: #000;
                      color: #ffa500;
                      text-align: center;
                      padding: 50px;
                  }
                  h1 {
                      color: #ffa500;
                  }
                  form {
                      background-color: rgba(255, 255, 255, 0.1);
                      padding: 20px;
                      border-radius: 8px;
                      box-shadow: 0 0 10px 0 rgba(255,165,0,0.6);
                      text-align: center;
                      border: 2px solid #ff8c00; /* Dim orange border */
                      margin: auto;
                      width: 500px;
                  }
                  .button {
                      background-color: #ff8c00;
                      color: white;
                      padding: 15px 20px;
                      margin: 10px 0;
                      border: none;
                      border-radius: 25px;
                      cursor: pointer;
                      transition: all 0.3s ease-in-out;
                      box-shadow: 0 4px 8px 0 rgba(255,140,0,0.5);
                  }
                  .button:hover {
                      opacity: 0.7;
                      background-color: #ffa500;
                      box-shadow: 0 8px 16px 0 rgba(255,165,0,0.8);
                  }
                  label {
                      margin-right: 10px;
                      display: block;
                      color: #fff;
                  }
              </style>
          </head>
          <body>
              <h1>Welcome to the Bajaj Expert System</h1>
              <form method="post">
                  <label><input type="checkbox" name="symptoms" value="not_starting"> Not Starting</label>
                  <label><input type="checkbox" name="symptoms" value="battery_low"> Battery Low</label>
                  <label><input type="checkbox" name="symptoms" value="battery_ok"> Battery OK</label>
                  <label><input type="checkbox" name="symptoms" value="starting"> Starting</label>
                  <label><input type="checkbox" name="symptoms" value="stalls_frequently"> Stalls Frequently</label>
                  <label><input type="checkbox" name="symptoms" value="poor_acceleration"> Poor Acceleration</label>
                  <label><input type="checkbox" name="symptoms" value="unusual_noises"> Unusual Noises</label>
                  <label><input type="checkbox" name="symptoms" value="engine_overheating"> Engine Overheating</label>
                  <label><input type="checkbox" name="symptoms" value="smoke_exhaust"> Smoke from Exhaust</label>
                  <label><input type="checkbox" name="symptoms" value="vibrating_excessively"> Vibrating Excessively</label>
                  <label><input type="checkbox" name="symptoms" value="lights_dim"> Lights Dim</label>
                  <label><input type="checkbox" name="symptoms" value="clicking_sound"> Clicking Sound</label>
                  <label><input type="checkbox" name="symptoms" value="leaking_fluids"> Leaking Fluids</label>
                  <label><input type="checkbox" name="symptoms" value="brake_issues"> Brake Issues</label>
                  <input type="submit" value="Diagnose" class="button">
              </form>
          </body>
          </html>
      """)
  
  if __name__ == "__main__":
      app.run(debug=True)
  
      `,
      language: "python"
    },
    Knowledge: {
      code: `class BajajExpertSystem:
      def __init__(self):
          # Initialize rules with conditions and conclusions
          self.rules = [
              {"conditions": ["not_starting", "battery_low"], "conclusion": "Charge the battery."},
              {"conditions": ["not_starting", "battery_ok"], "conclusion": "Check the starter motor."},
              {"conditions": ["starting", "stalls_frequently"], "conclusion": "Check the fuel supply."},
              {"conditions": ["starting", "poor_acceleration"], "conclusion": "Check the air filter."},
              {"conditions": ["starting", "unusual_noises"], "conclusion": "Check the engine."},
          ]
      `,
      language: "python"
    },
    infrence: {
      code: `def diagnose(self, symptoms):
        # Loop through each rule to find a match with the provided symptoms
        for rule in self.rules:
            if all(condition in symptoms for condition in rule["conditions"]):
                return rule["conclusion"]
        return "No diagnosis found. Please consult a professional mechanic."
      `,
      language: "python"
    },
    flask: {
      code: `from flask import Flask, request, render_template_string
  
    # Initialize the Flask application
    app = Flask(__name__)
      `,
      language: "python"
    },
    routes: {
      code: `@app.route('/', methods=['GET', 'POST'])
      def home():
          # Create an instance of the expert system
          system = BajajExpertSystem()
          if request.method == 'POST':
              # Get the list of symptoms from the form submission
              symptoms = request.form.getlist('symptoms')
              # Use the expert system to diagnose based on the symptoms
              diagnosis = system.diagnose(symptoms)
              # Render the diagnosis result
              return render_template_string("""
              <!DOCTYPE html>
              <html>
              <head>
                  <style>
                      body {
                          font-family: Arial, sans-serif;
                          background-color: #f4f4f4;
                          text-align: center;
                          padding: 50px;
                      }
                      h1 {
                          color: #333;
                      }
                      p {
                          color: #666;
                      }
                      .button {
                          background-color: #4CAF50;
                          color: white;
                          padding: 14px 20px;
                          margin: 8px 0;
                          border: none;
                          cursor: pointer;
                          transition: all 0.3s ease 0s;
                      }
                      .button:hover {
                          background-color: #45a049;
                      }
                      form {
                          background-color: #ffffff;
                          padding: 20px;
                          border-radius: 8px;
                          box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                      }
                      label {
                          margin-right: 10px;
                      }
                  </style>
              </head>
              <body>
                  <h1>Diagnosis Result</h1>
                  <p>{{ diagnosis }}</p>
                  <a href="/" class="button">Back</a>
              </body>
              </html>
          """, diagnosis=diagnosis)
          # Render the initial form for users to input symptoms
          return render_template_string("""
              <!DOCTYPE html>
              <html>
              <head>
                  <style>
                      body {
                          font-family: 'Arial', sans-serif;
                          background-color: #f4f4f4;
                          text-align: center;
                          padding: 50px;
                      }
                      h1 {
                          color: #333;
                      }
                      form {
                          background-color: #ffffff;
                          padding: 20px;
                          border-radius: 8px;
                          box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                      }
                      .button {
                          background-color: #008CBA;
                          color: white;
                          padding: 15px 20px;
                          margin: 10px 0;
                          border: none;
                          cursor: pointer;
                          transition: opacity 0.3s ease-in-out;
                      }
                      .button:hover {
                          opacity: 0.7;
                      }
                      label {
                          margin-right: 10px;
                      }
                  </style>
              </head>
              <body>
                  <h1>Welcome to the Bajaj Expert System</h1>
                  <form method="post">
                      <label><input type="checkbox" name="symptoms" value="not_starting"> Not Starting</label><br>
                      <label><input type="checkbox" name="symptoms" value="battery_low"> Battery Low</label><br>
                      <label><input type="checkbox" name="symptoms" value="battery_ok"> Battery OK</label><br>
                      <label><input type="checkbox" name="symptoms" value="starting"> Starting</label><br>
                      <label><input type="checkbox" name="symptoms" value="stalls_frequently"> Stalls Frequently</label><br>
                      <label><input type="checkbox" name="symptoms" value="poor_acceleration"> Poor Acceleration</label><br>
                      <label><input type="checkbox" name="symptoms" value="unusual_noises"> Unusual Noises</label><br>
                      <input type="submit" value="Diagnose" class="button">
                  </form>
              </body>
              </html>
          """)
      `,
      language: "python"
    },
    html: {
      code: `return render_template_string("""
      <!DOCTYPE html>
      <html>
      <head>
          <style>
              body {
                  font-family: 'Arial', sans-serif;
                  background-color: #f4f4f4;
                  text-align: center;
                  padding: 50px;
              }
              h1 {
                  color: #333;
              }
              form {
                  background-color: #ffffff;
                  padding: 20px;
                  border-radius: 8px;
                  box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
              }
              .button {
                  background-color: #008CBA;
                  color: white;
                  padding: 15px 20px;
                  margin: 10px 0;
                  border: none;
                  cursor: pointer;
                  transition: opacity 0.3s ease-in-out;
              }
              .button:hover {
                  opacity: 0.7;
              }
              label {
                  margin-right: 10px;
              }
          </style>
      </head>
      <body>
          <h1>Welcome to the Bajaj Expert System</h1>
          <form method="post">
              <label><input type="checkbox" name="symptoms" value="not_starting"> Not Starting</label><br>
              <label><input type="checkbox" name="symptoms" value="battery_low"> Battery Low</label><br>
              <label><input type="checkbox" name="symptoms" value="battery_ok"> Battery OK</label><br>
              <label><input type="checkbox" name="symptoms" value="starting"> Starting</label><br>
              <label><input type="checkbox" name="symptoms" value="stalls_frequently"> Stalls Frequently</label><br>
              <label><input type="checkbox" name="symptoms" value="poor_acceleration"> Poor Acceleration</label><br>
              <label><input type="checkbox" name="symptoms" value="unusual_noises"> Unusual Noises</label><br>
              <input type="submit" value="Diagnose" class="button">
          </form>
      </body>
      </html>
  """)
      `,
      language: "python"
    },
    run: {
      code: `if __name__ == "__main__":
          app.run(debug=True)
      `,
      language: "python"
    }
  };
  
  export default codeSections;
  