from flask import Flask, request, jsonify
from summarizer import generate_summary_for_text  # Import your summarization function

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()  # Get the data from the frontend
    text = data['text']  # Extract the text from the request
    summary = generate_summary_for_text(text, con_length=1)  # Call the summarization function
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
