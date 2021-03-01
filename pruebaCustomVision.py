from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import time

# Replace with valid values
ENDPOINT = "https://customvisionmosca.cognitiveservices.azure.com/"
training_key = "6d86fa9721694154b86157dda30a54af"
prediction_key = "3781d90957144cfb8cf52b7a4423492f"
prediction_resource_id = "/subscriptions/d7b2c753-e443-4211-9592-14e43d67e24b/resourceGroups/customVision-mosca/providers/Microsoft.CognitiveServices/accounts/customVisionMosca"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

publish_iteration_name = "classifyModel"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# Recuperando el proyecto.
print ("Creando proyecto...")
project = trainer.create_project("clasificacionMosca_2")

# Creación de las etiquetas.
mosca_tag = trainer.create_tag(project.id, "Mosca")
gusano_tag = trainer.create_tag(project.id, "Gusano")
palomilla_tag = trainer.create_tag(project.id, "Palomilla")
pulgon_tag = trainer.create_tag(project.id, "Pulgon")

base_image_location = "D:\Cursos/customVision-python/"
print("Adding images...")

image_list = []

# Lectura de las imagenes de mosca blanca.
for image_num in range(1, 15):
    file_name = "mosca{}.jpg".format(image_num)
    with open(base_image_location + "img_etiquetas/mosca/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[mosca_tag.id]))

# Lectura de las imagenes de gusanos.
for image_num in range(1, 12):
    file_name = "gusano_{}.jpg".format(image_num)
    with open(base_image_location + "img_etiquetas/gusano/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[gusano_tag.id]))

# Lectura de las imagenes de palomillas.
for image_num in range(1, 12):
    file_name = "palomilla_{}.jpg".format(image_num)
    with open(base_image_location + "img_etiquetas/palomilla/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[palomilla_tag.id]))

# Lectura de las imagenes del pulgones.
for image_num in range(1, 12):
    file_name = "pulgon_{}.jpg".format(image_num)
    with open(base_image_location + "img_etiquetas/pulgon/" + file_name, "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[pulgon_tag.id]))

upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
if not upload_result.is_batch_successful:
    print("Image batch upload failed.")
    for image in upload_result.images:
        print("Image status: ", image.status)
    #exit(-1)

print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

with open(base_image_location + "img_test/test_imagen.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        project.id, publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))
