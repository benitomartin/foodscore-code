# **THE BUSINESS MODEL** :business_suit_levitating:

## :pushpin: **Pain Points**

In recent years, food recognition using deep learning models has become an active research area. 
However, there are still several pain points that researchers and developers need to address in order to improve the performance of these models. Here are some of them:

- <ins>Limited availability and quality of food datasets</ins>: Collecting and labelling large-scale datasets of diverse food images is time-consuming, expensive, and requires a lot of effort, which makes it challenging to train models that generalize well to unseen data.

- <ins>Variability in food appearance</ins>: Food can vary widely in appearance. This variability can lead to challenges in developing accurate models that can recognize different types of food.

- <ins>Overlapping of food items</ins>: In real-world images, food items may overlap or be partially occluded by other objects, which makes difficult for deep learning models to accurately recognize them and their boundaries.

- <ins>Computational complexity</ins>: Deep learning models for food recognition typically require large amounts of computational resources and processing power, which can be expensive and time-consuming.

## :thinking: **The Idea**

Worldwide overweight in the adult population has increased from 39% to 45% in the past 5 years and a healthy diet prevents overweight and related health problems. Additionally, the Europe Diet and Nutrition Apps Market is expected to account for USD 4,580 Million by 2028. With these insights, developing a model to classify a food item within an image and return its nutrition data will help to track our diet's nutritional composition. 

## :spiral_calendar: **The Timeline**

The whole project was developed in 2 weeks by a team of 4 data scientists. 

![Screenshot timeline](https://user-images.githubusercontent.com/116911431/230025026-09321fdf-2e6c-452f-92a6-e2b2197ce9a7.png)

## :rice: **The Data**

The Dataset used was the [Food100](http://foodcam.mobi/dataset100.html) from the Food Recognition Research Group (University of Electro-Communications, Tokyo, Japan). This Dataset contains a mixture of Western and Japanese food, 100 different classes, and around 12740 photos, including 1174 multiple-food photos.

![My Remote Image](https://user-images.githubusercontent.com/116911431/229874089-2a3d3f32-a45f-453d-8eb1-a6270e4c64a6.jpg)

Additionally, the nutrition facts were extracted from the database [FoodData Central](https://fdc.nal.usda.gov/index.html)
 of the U.S. Department of Agriculture.
 
## :atom: **The Model**

#### :one: **Data Preparation**

The dataset is well organized, and it can be easily split into train, validation, and test set using `image_dataset_from_directory`.

#### :two: **Data Augmentation**

Due to some unbalanced classes, data augmentation was implemented. The following layers from Tensorflow Keras were used:
- `RandomFlip`
- `RandomRotation`
- `RandomBrightness`
- `RandomContrast`
- `RandomZoom`

Additionally, a layer called `RandomBlurHue` was created to adjust the color hue of RGB images by a random factor and perform Gaussian blur on the images.

#### :three: **Data Training and Evaluation**

Transfer learning using `ResNet152`was implemented. This model has convolutional weights that are trained on ImageNet data, and compared to the other Keras Models, it had the best performance.


#### :four: **Deployment**

For deployment, the following steps were taken:

👉 create a prediction API using FastAPI

👉 create a Docker image containing the environment required to run the code of our API

👉 push this image to Google Cloud Run so that it runs inside a Docker container that allows people all over the world to use it.

## :clapper: **The Business Solution :arrow_right: Website**

As a business solution a Website was created using Streamlit. For that, an additional [repository](https://github.com/benitomartin/foodscore-app) was created and is available for everyone. Instructions on how to use the Website can be found in that repository.

![Screenshot App](https://user-images.githubusercontent.com/116911431/229893707-e95bf9ff-0d50-4d12-a6b9-cdb88ffc54e6.png)

## **Conclusion** :end:

After training a deep learning model for food recognition, we can conclude that the model is capable of accurately identifying different types of food items based on their visual features. The model has been trained on a large dataset of food images, and through the use of advanced techniques, it has learned to recognize the unique features of different food categories.

However, as with any machine learning model, there is always room for improvement. The accuracy of the model can be further improved by fine-tuning the hyperparameters, increasing the size of the training dataset, and incorporating additional data augmentation techniques.

In conclusion, the deep learning model for food recognition has shown great promise in accurately identifying different food items and has the potential to be applied in various real-world scenarios. Further research and development can help to improve the accuracy and robustness of the model, making it even more useful in the future.

## :question: **How to use this repository** 

First, you need to clone the repository and install the requirements by running `pip install -r requirements.txt`. This allows the installation of all required packages. Then follow these steps:

- Create an additional directory `raw_data/UECFOOD100/` and save there the Food100 dataset
- Create an additional directory `raw_data/model` where the model will be saved
- Run main.py to train the model, get the weights, and the nutrients. The weights will be saved in the `raw_data/model`
- Create the Docker image and push it to Google Cloud Run
- In the Website [repository](https://github.com/benitomartin/foodscore-app), under app.py, change the url under line 96, for your own url.

