## Age-Gender-Prediction-and-Emotion-recognition-using-CNN
This project has two models, one for age-gender prediction using wide resnet architecture and the other model is trained for emotion recognition using conventional CNN architecture. The dataset was obtained from [IMDb-WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) for age-gender classification and [Fer2013](https://www.kaggle.com/deadskull7/fer2013) dataset for emotion recognition.

### Training Age-Gender model
First, download the imdb-wiki dataset in train/data/ dir, then clean the data and serialize labels by:
```
python create_db.py --db imdb
```
Then, start the training by:
```
python train.py
```

### Demonstration 
Put the trained models in models dir, then run the following for webcam demo:
```
python age-gender-emotion.py
```
