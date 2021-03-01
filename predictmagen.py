from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import time

ENDPOINT = "https://customvisionmosca.cognitiveservices.azure.com/"
training_key = "6d86fa9721694154b86157dda30a54af"
prediction_key = "3781d90957144cfb8cf52b7a4423492f"
prediction_resource_id = "/subscriptions/d7b2c753-e443-4211-9592-14e43d67e24b/resourceGroups/customVision-mosca/providers/Microsoft.CognitiveServices/accounts/customVisionMosca"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

base_image_location = "D:\Cursos/customVision-python/"
publish_iteration_name = "Iteration 1"

with open(base_image_location + "img_test/test_imagen.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        "a5817b55-5218-40d5-981a-63e5318faf02", publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))