from flask import Flask, jsonify, request
from flask_cors import CORS
from chat import answer

app = Flask(__name__)
CORS(app)

@app.get("/")
def index():
    return jsonify({"status":"ok","message":"Kevin McCarthy paper assistant is running."})

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query","")).strip()

    if not query:
        return jsonify({"error":"No query provided."}),400

    if len(query) > 1000:
        return jsonify({"error":"Question is too long."}),400

    try:
        text,results = answer(query)
        sources = [{"label":f"SOURCE {n}","paper":r["paper"],"page_start":r["page_start"],"page_end":r["page_end"],"similarity":round(r["score"],3)} for n,r in enumerate(results,1)]
        return jsonify({"answer":text,"sources":sources})
    except Exception as exc:
        print(f"Chat error: {exc}")
        return jsonify({"error":"The assistant could not answer the question."}),500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
