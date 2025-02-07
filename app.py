from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# Store the history of summaries
history = []

@app.route('/')
def index():
    return render_template('index.html', history=history)

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get("text")
    summary = summarizer(text)[0]['summary_text']
    history.append({'original': text, 'summary': summary})
    return render_template('index.html', summary=summary, history=history)

if __name__ == "__main__":
    app.run(debug=True)
