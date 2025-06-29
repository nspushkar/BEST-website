const codeSections = {
    Step1: {
      code: `# Generate Training Data
      import pandas as pd
      import numpy as np
    
      # Parameters
      n_samples = 1000
    
      # Generate data
      np.random.seed(42)
      temperature = np.random.normal(loc=25, scale=5, size=n_samples)
      pressure = np.random.normal(loc=1, scale=0.2, size=n_samples)
      vibration = np.random.normal(loc=0.01, scale=0.005, size=n_samples)
    
      # Classify data as healthy or unhealthy
      def classify(temperature, pressure, vibration):
          if temperature > 30 or pressure > 1.2 or vibration > 0.02:
              return 'unhealthy'
          else:
              return 'healthy'
    
      data = pd.DataFrame({
          'temperature': temperature,
          'pressure': pressure,
          'vibration': vibration
      })
      data['status'] = data.apply(lambda row: classify(row['temperature'], row['pressure'], row['vibration']), axis=1)
      `,
      language: 'python'
    },
      
    Step2: {
      code: `# Save Training Data
      data.to_csv('training_data.csv', index=False)
      print("Training data generated and saved to 'training_data.csv'")
      `,
      language: 'python'
    },
      
    Step3: {
      code: `# Load and Preprocess Data
      data = pd.read_csv('training_data.csv')
      data.head()
    
      # Extract features and labels
      X = data[['temperature', 'pressure', 'vibration']]
      y = data['status']
    
      # Convert labels to numerical values
      y = y.map({'healthy': 0, 'unhealthy': 1})
      `,
      language: 'python'
    },
      
    Step4: {
      code: `# Split Data into Training and Testing Sets
      from sklearn.model_selection import train_test_split
    
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      `,
      language: 'python'
    },
      
    Step5: {
      code: `# Train the Model
      from sklearn.ensemble import RandomForestClassifier
    
      # Initialize the model
      model = RandomForestClassifier(n_estimators=100, random_state=42)
    
      # Train the model
      model.fit(X_train, y_train)
      `,
      language: 'python'
    },
      
    Step6: {
      code: `# Evaluate the Model
      from sklearn.metrics import accuracy_score
    
      # Evaluate the model
      y_pred = model.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      print(f'Model Accuracy: {accuracy:.2f}')
      `,
      language: 'python'
    },
      
    Step7: {
      code: `# Save the Model
      import joblib
    
      # Save the model to a file
      joblib.dump(model, 'equipment_monitoring_model.pkl')
      print("Model saved to 'equipment_monitoring_model.pkl'")
      `,
      language: 'python'
    },
      
    Step8: {
      code: `# Create the Dashboard
      import dash
      import dash_core_components as dcc
      import dash_html_components as html
      import plotly.graph_objs as go
      from dash.dependencies import Input, Output
      from collections import deque
      import pandas as pd
      import numpy as np
      import joblib
      import time
    
      # Load the trained model
      model = joblib.load('equipment_monitoring_model.pkl')
    
      # Initialize the app
      app = dash.Dash(__name__)
    
      # Create a deque to hold the real-time data
      window_size = 30  # Define the size of the sliding window (30 seconds)
      data_deque = deque(maxlen=window_size)
      error_timestamps = deque(maxlen=1000)  # Keep track of error timestamps
      `,
      language: 'python'
    },
      
    Step9: {
      code: `# Real-time Data Generator
      def real_time_data_generator():
          while True:
              temperature = np.random.normal(loc=25, scale=5)
              pressure = np.random.normal(loc=1, scale=0.2)
              vibration = np.random.normal(loc=0.01, scale=0.005)
              yield {'temperature': temperature, 'pressure': pressure, 'vibration': vibration}
    
      data_gen = real_time_data_generator()
      `,
      language: 'python'
    },
      
    Step10: {
      code: `# Update Graphs
      @app.callback(
          Output('temperature-graph', 'figure'),
          [Input('interval-component', 'n_intervals')]
      )
      def update_temperature_graph(n):
          next(data_gen)  # Update the deque with new data
          df = pd.DataFrame(data_deque)
          if df.empty:
              return go.Figure()  # Return an empty figure if no data
    
          z = df['temperature']
          figure = {
              'data': [
                  go.Scatter(
                      x=list(range(n-len(df), n)),
                      y=df['temperature'],
                      mode='lines+markers',
                      marker=dict(
                          size=10,
                          color=z,
                          colorscale='RdBu_r',  # Inverted RdBu colorscale
                          showscale=True,
                          colorbar=dict(title='Temperature')
                      ),
                      line=dict(
                          color='black',
                          width=2
                      ),
                      name='Temperature'
                  )
              ],
              'layout': go.Layout(
                  title='Temperature Over Time',
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='Temperature (°C)'),
                  xaxis_range=[max(0, n-window_size), n]  # Sliding window
              )
          }
          return figure
    
      # Similar code for pressure and vibration graphs
      `,
      language: 'python'
    },
      
    Step11: {
      code: `# Update Status Indicator
      def get_status_style(status):
          if status == 0:
              return {
                  'backgroundColor': '#98FB98',  # pastel green
                  'color': '#006400',  # dark green text
                  'padding': '20px',
                  'borderRadius': '10px',
                  'width': '96.3%',
                  'textAlign': 'center',
                  'margin': '0'
              }
          else:
              return {
                  'backgroundColor': '#FFCCCB',  # pastel red
                  'color': '#8B0000',  # dark red text
                  'padding': '20px',
                  'borderRadius': '10px',
                  'width': '96.3%',
                  'textAlign': 'center',
                  'margin': '0'
              }
    
      @app.callback(
          Output('status-container', 'children'),
          [Input('interval-component', 'n_intervals')]
      )
      def update_status_indicator(n):
          df = pd.DataFrame(data_deque)
          if not df.empty:
              latest_data = df[['temperature', 'pressure', 'vibration']].iloc[-1].to_frame().T
              current_status = model.predict(latest_data)[0]
              status_text = get_status_color_and_text(current_status)
              status_style = get_status_style(current_status)
              
              if current_status == 1:
                  error_timestamps.append(time.time())
              
              return html.Div(f'Current Status: {status_text}', style=status_style)
          else:
              return html.Div('No Data', style={'backgroundColor': 'black', 'color': 'white', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '100%', 'margin': '0'})
      `,
      language: 'python'
    },
      
    Step12: {
      code: `# Run the Dashboard
      if __name__ == '__main__':
          app.run_server(debug=True)
      `,
      language: 'python'
    },
  
    full: {
      code: `# Full Implementation
      import pandas as pd
      import numpy as np
      from sklearn.model_selection import train_test_split
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.metrics import accuracy_score
      import joblib
      import dash
      import dash_core_components as dcc
      import dash_html_components as html
      import plotly.graph_objs as go
      from dash.dependencies import Input, Output
      from collections import deque
      import time
  
      # Generate Training Data
      n_samples = 1000
      np.random.seed(42)
      temperature = np.random.normal(loc=25, scale=5, size=n_samples)
      pressure = np.random.normal(loc=1, scale=0.2, size=n_samples)
      vibration = np.random.normal(loc=0.01, scale=0.005, size=n_samples)
    
      def classify(temperature, pressure, vibration):
          if temperature > 30 or pressure > 1.2 or vibration > 0.02:
              return 'unhealthy'
          else:
              return 'healthy'
    
      data = pd.DataFrame({
          'temperature': temperature,
          'pressure': pressure,
          'vibration': vibration
      })
      data['status'] = data.apply(lambda row: classify(row['temperature'], row['pressure'], row['vibration']), axis=1)
      data.to_csv('training_data.csv', index=False)
  
      # Load and Preprocess Data
      data = pd.read_csv('training_data.csv')
      X = data[['temperature', 'pressure', 'vibration']]
      y = data['status'].map({'healthy': 0, 'unhealthy': 1})
  
      # Split Data into Training and Testing Sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
      # Train the Model
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X_train, y_train)
  
      # Evaluate the Model
      y_pred = model.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      print(f'Model Accuracy: {accuracy:.2f}')
  
      # Save the Model
      joblib.dump(model, 'equipment_monitoring_model.pkl')
      
      # Create the Dashboard
      model = joblib.load('equipment_monitoring_model.pkl')
      app = dash.Dash(__name__)
      window_size = 30  
      data_deque = deque(maxlen=window_size)
      error_timestamps = deque(maxlen=1000) 
  
      def real_time_data_generator():
          while True:
              temperature = np.random.normal(loc=25, scale=5)
              pressure = np.random.normal(loc=1, scale=0.2)
              vibration = np.random.normal(loc=0.01, scale=0.005)
              yield {'temperature': temperature, 'pressure': pressure, 'vibration': vibration}
  
      data_gen = real_time_data_generator()
  
      app.layout = html.Div(children=[
          html.H1(children='Real-Time Industrial Equipment Monitoring Dashboard'),
          dcc.Graph(id='temperature-graph'),
          dcc.Graph(id='pressure-graph'),
          dcc.Graph(id='vibration-graph'),
          html.Div(id='status-container', style={'textAlign': 'center', 'marginTop': '20px', 'width': '100%', 'padding': '0'}),
          html.Div(id='additional-info', style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px', 'width': '100%'}),
          dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
      ])
  
      @app.callback(
          Output('temperature-graph', 'figure'),
          [Input('interval-component', 'n_intervals')]
      )
      def update_temperature_graph(n):
          next(data_gen)
          df = pd.DataFrame(data_deque)
          if df.empty:
              return go.Figure() 
  
          z = df['temperature']
          figure = {
              'data': [
                  go.Scatter(
                      x=list(range(n-len(df), n)),
                      y=df['temperature'],
                      mode='lines+markers',
                      marker=dict(
                          size=10,
                          color=z,
                          colorscale='RdBu_r', 
                          showscale=True,
                          colorbar=dict(title='Temperature')
                      ),
                      line=dict(
                          color='black',
                          width=2
                      ),
                      name='Temperature'
                  )
              ],
              'layout': go.Layout(
                  title='Temperature Over Time',
                  xaxis=dict(title='Time'),
                  yaxis=dict(title='Temperature (°C)'),
                  xaxis_range=[max(0, n-window_size), n] 
              )
          }
          return figure
  
      # Similar code for pressure and vibration graphs
  
      def get_status_style(status):
          if status == 0:
              return {
                  'backgroundColor': '#98FB98',
                  'color': '#006400', 
                  'padding': '20px',
                  'borderRadius': '10px',
                  'width': '96.3%',
                  'textAlign': 'center',
                  'margin': '0'
              }
          else:
              return {
                  'backgroundColor': '#FFCCCB', 
                  'color': '#8B0000', 
                  'padding': '20px',
                  'borderRadius': '10px',
                  'width': '96.3%',
                  'textAlign': 'center',
                  'margin': '0'
              }
  
      @app.callback(
          Output('status-container', 'children'),
          [Input('interval-component', 'n_intervals')]
      )
      def update_status_indicator(n):
          df = pd.DataFrame(data_deque)
          if not df.empty:
              latest_data = df[['temperature', 'pressure', 'vibration']].iloc[-1].to_frame().T
              current_status = model.predict(latest_data)[0]
              status_text = get_status_color_and_text(current_status)
              status_style = get_status_style(current_status)
              
              if current_status == 1:
                  error_timestamps.append(time.time())
              
              return html.Div(f'Current Status: {status_text}', style=status_style)
          else:
              return html.Div('No Data', style={'backgroundColor': 'black', 'color': 'white', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center', 'width': '100%', 'margin': '0'})
  
      if __name__ == '__main__':
          app.run_server(debug=True)
      `,
      language: 'python'
    },
  };
  
  export default codeSections;
  