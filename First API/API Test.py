## API Test
## Script from: https://programminghistorian.org/en/lessons/creating-apis-with-python-and-flask
## To Test: open with WebBrowser: http://127.0.0.1:5000/


import flask
from flask import request


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route("/nome", methods=['GET'])
def nome():
    nome = request.args['nome']
    return ("<h1>"+nome+"</h1>")
    
app.run()
 


