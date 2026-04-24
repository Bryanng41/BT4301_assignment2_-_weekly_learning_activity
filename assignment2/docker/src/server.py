from connexion import FlaskApp
from flask import jsonify

app = FlaskApp(__name__)
app.add_api("predict.yml")

@app.app.get("/api/health")
def health():
    return jsonify(status="ok")

# WSGI entry (e.g. Gunicorn in Docker).
wsgi = app.app

if __name__ == "__main__":
    app.app.run(
        host="0.0.0.0",
        port=5000,
        use_reloader=False,
        threaded=True,
    )
