from connexion import FlaskApp
from flask import Response, jsonify

app = FlaskApp(__name__)
app.add_api("predict.yml")

# Root and health on the underlying Flask app: Connexion can mishandle plain
# string bodies / MIME for GET / in some versions, which surfaces in the browser
# as ERR_EMPTY_RESPONSE (-324).
@app.app.get("/")
def index_root():
    html = (
        "<h1>Purchase prediction API (customer × product)</h1>"
        "<p>GET <code>/api/purchase?customer_id=...&amp;product_id=...</code> "
        "(see <a href=\"/ui/\">Swagger UI</a>).</p>"
    )
    return Response(html, mimetype="text/html")


@app.app.get("/api/health")
def health():
    return jsonify(status="ok")


# WSGI entry for production-style servers (Gunicorn in Docker). Binds 0.0.0.0:5000 reliably for
# host port mapping; the Werkzeug dev server can return empty replies on some Docker Desktop Mac setups.
wsgi = app.app


if __name__ == "__main__":
    # Local dev: same Flask app, no Connexion FlaskApp.run() quirks; optional—Compose uses gunicorn.
    app.app.run(
        host="0.0.0.0",
        port=5000,
        use_reloader=False,
        threaded=True,
    )
