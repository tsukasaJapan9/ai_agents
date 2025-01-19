from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # templates フォルダ内の index.html を提供

if __name__ == '__main__':
    app.run(debug=True)