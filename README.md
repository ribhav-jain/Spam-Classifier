# Spam-Classifier
ML-powered Flask app to perform spam classification of SMS messages using TF-IDF vectorization.

### Installation
#### Prerequisites
Make sure that you have the following:

Python 3+ and pip (which comes with Python 3+)

Run the following command on the root directory of this project:

**Optional commands to create virual environment**:
1. Create virtual environment: `python -m venv Spam-Classifier`
2. Activate virual environment: `Spam-Classifier\Scripts\activate`

**Command to start server**:
1. Install Dependency: `pip install -r requirements.txt`
2. `flask run`

This will start flask based spam classifier server in **production mode**.

To access home page browse to `http://localhost:5000`

**To start python notebook from within python virtual environment follow below commands:**
1. `python -m pip install jupyter`
2. `jupyter notebook`
3. Open `Data Analysis and Modelling.ipynb` from browser

### Web app routes:
1. `localhost:5000\train`  will load dataset and create vocabulary, train Naive Bayes model using that.
2. `localhost:5000\` is the main page of the web app.
3. `localhost:5000\predict` will predict the class of text passed. This api will load trained model which were created from `localhost:5000\train` and use it for prediction.
