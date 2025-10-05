# Agmar LSTM Model

This project implements a Long Short-Term Memory (LSTM) model for predicting the modal price of a commodity based on the historical data provided in the "arecanut.csv" dataset.

## Project Overview

The main steps of the project are as follows:

1.  **Data Preprocessing**: The dataset is preprocessed by converting the "Price Date" column to datetime format. The relevant columns are selected, and the data is scaled using the MinMaxScaler.

2.  **Model Building**: A sequential LSTM model is constructed using Keras. It consists of two LSTM layers and a dense output layer.

3.  **Model Training**: The model is compiled with the Adam optimizer and trained on the preprocessed data.

4.  **Model Evaluation**: The trained model is evaluated using the testing data, and the Mean Absolute Error (MAE) and R-squared are calculated as evaluation metrics.

5.  **Prediction and Visualization**: The model is utilized to make predictions on the testing data, and the results are visualized by comparing the actual and predicted values.

6.  **Model Saving**: The trained LSTM model is saved to a file named "model.h5" and the scaler is saved to "scaler.gz" for future use in a web application.

## Prerequisites

To run the project, you'll need:

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* tensorflow
* flask
* flask_cors
* joblib

## Usage

1.  **Install Dependencies**:
    ```shell
    pip install pandas numpy scikit-learn matplotlib seaborn tensorflow flask flask_cors joblib
    ```

2.  **Train the Model**:
    Run the `agmar_lstm_model.py` script to train the model and generate `model.h5` and `scaler.gz`.
    ```shell
    python agmar_lstm_model.py
    ```

3.  **Run the Web App**:
    Start the Flask server by running `app.py`.
    ```shell
    python app.py
    ```

4.  **View the Predictor**:
    Open the `index.html` file in your web browser. You should see the price predictor interface. Click the "Predict Tomorrow's Price" button to get a prediction.

## References

* [pandas Documentation](https://pandas.pydata.org/docs/)
* [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
* [matplotlib Documentation](https://matplotlib.org/contents.html)
* [seaborn Documentation](https://seaborn.pydata.org/tutorial.html)
* [TensorFlow Documentation](https://www.tensorflow.org/api_docs)