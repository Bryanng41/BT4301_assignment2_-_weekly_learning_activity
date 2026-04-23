from connexion import FlaskApp



app = FlaskApp(__name__)
app.add_api('iris.yml')



@app.route('/')
def index():

	return '''
	<h1>Welcome to the Iris Flower Decision Tree Classifier</h1>
    <p>Use the REST API endpoint at /api/iris to classify the species of new flowers.</p>
    '''



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
