const codeSections = {
    full: {
      code: `!pip install spacy
  !python -m spacy download en_core_web_sm
  !pip install Flask
  
  import spacy
  from flask import Flask, request, jsonify
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.pipeline import make_pipeline
  
  # Load the spaCy model
  nlp = spacy.load('en_core_web_sm')
  
  # Define the set of training data
  queries = [
      "What are your business hours?",
      "How do I reset my password?",
      "Where is my order?",
      "Can I return an item?",
      "What is the status of my ticket?",
      "How do I contact support?"
  ]
  
  responses = [
      "Our business hours are 9 AM to 5 PM, Monday to Friday.",
      "To reset your password, click on 'Forgot Password' on the login page.",
      "You can check your order status by logging into your account and visiting the 'Orders' section.",
      "To return an item, please visit our returns page and follow the instructions.",
      "Your ticket status can be checked in the 'Support' section after logging in.",
      "You can contact support by emailing support@example.com or calling 123-456-7890."
  ]
  
  # Create and train a simple model using scikit-learn
  model = make_pipeline(CountVectorizer(), MultinomialNB())
  model.fit(queries, responses)
  
  # Initialize Flask app
  app = Flask(__name__)
  
  # Define the route for the chatbot
  @app.route('/chatbot', methods=['POST'])
  def chatbot():
      # Get the user's query from the request
      query = request.json.get('query')
  
      # Predict the response using the trained model
      response = model.predict([query])[0]
  
      # Return the response as JSON
      return jsonify({"response": response})
  
  # Function to run the app in a separate thread
  def run_app():
      try:
          app.run(debug=True, port=5002, use_reloader=False)
      except Exception as e:
          print(f"Error: {e}")
  
  # Run the Flask app in a separate thread
  from threading import Thread
  flask_thread = Thread(target=run_app)
  flask_thread.start()
  
  import requests
  
  def get_chatbot_response(query):
      # Define the URL for the chatbot
      url = "http://127.0.0.1:5002/chatbot"
  
      # Send the POST request with the user query
      response = requests.post(url, json={"query": query})
  
      # Get the JSON response and extract the message
      return response.json().get("response")
  
  # Function to test chatbot with a given query
  def test_chatbot(query):
      # Get the chatbot response
      response = get_chatbot_response(query)
  
      # Print the response
      print("User query:", query)
      print("Chatbot response:", response)
  
  # Example query for testing
  test_query = "What are your business hours?"
  test_chatbot(test_query)
  
  test_chatbot("How do I reset my password?")
  test_chatbot("Where is my order?")
      `,
      language: "python",
    },
    SpaCy: {
      code: `import spacy
  from flask import Flask, request, jsonify
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.pipeline import make_pipeline
  
  # Load the spaCy model
  nlp = spacy.load('en_core_web_sm')
  
  # Define the set of training data
  queries = [
      "What are your business hours?",
      "How do I reset my password?",
      "Where is my order?",
      "Can I return an item?",
      "What is the status of my ticket?",
      "How do I contact support?"
  ]
  
  responses = [
      "Our business hours are 9 AM to 5 PM, Monday to Friday.",
      "To reset your password, click on 'Forgot Password' on the login page.",
      "You can check your order status by logging into your account and visiting the 'Orders' section.",
      "To return an item, please visit our returns page and follow the instructions.",
      "Your ticket status can be checked in the 'Support' section after logging in.",
      "You can contact support by emailing support@example.com or calling 123-456-7890."
  ]
  
  # Create and train a simple model using scikit-learn
  model = make_pipeline(CountVectorizer(), MultinomialNB())
  model.fit(queries, responses)
      `,
      language: "python",
    },
    Flask: {
      code: `app = Flask(__name__)
  
  # Define the route for the chatbot
  @app.route('/chatbot', methods=['POST'])
  def chatbot():
      # Get the user's query from the request
      query = request.json.get('query')
      
      # Predict the response using the trained model
      response = model.predict([query])[0]
  
      # Return the response as JSON
      return jsonify({"response": response})
  # Function to run the app in a separate thread
  def run_app():
      try:
          app.run(debug=True, port=5002, use_reloader=False)
      except Exception as e:
          print(f"Error: {e}")
  
  # Run the Flask app in a separate thread
  from threading import Thread
  flask_thread = Thread(target=run_app)
  flask_thread.start()
  
      `,
      language: "python",
    },
    RESTful: {
      code: `def get_chatbot_response(query):
      # Define the URL for the chatbot
      url = "http://127.0.0.1:5002/chatbot"
      
      # Send the POST request with the user query
      response = requests.post(url, json={"query": query})
      
      # Get the JSON response and extract the message
      return response.json().get("response")
      `,
      language: "python",
    },
  };
  
  export default codeSections;
