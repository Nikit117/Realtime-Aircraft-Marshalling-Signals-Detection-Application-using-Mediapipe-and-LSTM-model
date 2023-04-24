# Realtime-Aircraft-Marshalling-Signals-Detection-Application-using-Mediapipe-and-LSTM-model
This is a Realtime Aircraft Marshalling Signals Detection Application made using Mediapipe and LSTM model. It is made using tkinter and it classifies between 10 signals. It also has 5 features: to take screenshot, remove annotation, pause/play, change video source and change classification model.

## Requirements
To run this application, you need to have the following installed:

* Python 3.7 or higher
* mediapipe
* tensorflow
* opencv-python
* tkinter
* Installation
## You can install the required libraries using pip:

```python
pip install mediapipe tensorflow opencv-python tkinter
```

## Usage
To start the application, run the following command:
```python
python tkapp.py
```

Once the application starts, you will see a window with the webcam video feed. The application will start detecting the marshalling signals in real-time. You can take a screenshot by clicking on the `"Take Screenshot"` button. You can also remove the annotation by clicking on the `"Remove Annotation"` button. To pause/play the video, click on the `"Pause/Play"` button. To change the video source, click on the `"Change Video Source"` button.

## Model Training
The LSTM model used for classification was trained on a dataset of marshalling signals using the tensorflow library. The code for training the model is included in the `train.ipynb` notebook. You can use this notebook to train your own model using your own dataset.

## License
This application is licensed under the MIT License. You are free to use, modify, and distribute this software in any way you like. See the LICENSE file for more information.

## Credits
This application was made by Nikit Kashyap as a project for Mini project at Anna University.
