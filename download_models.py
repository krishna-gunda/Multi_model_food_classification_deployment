import gdown
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Model file IDs from Google Drive
models = {

    "vgg16_food_classification_model.keras":
    "1vyyYuUxVCj8q0g9W2yU5vy03uFq9Yjct",

    "food_classification_custom_Resnetmodel.keras":
    "1lc4ryZmSA29W8sFkH_vy29FvS22scLNf",

    "food_classification_custom_model.h5":
    "13FMgQHyfOK_lZG7KJbVwHDuNmhnwomlI"
}

# Download models if not already downloaded
for filename, file_id in models.items():

    output = f"models/{filename}"

    if not os.path.exists(output):

        url = f"https://drive.google.com/uc?id={file_id}"

        print(f"Downloading {filename}...")

        gdown.download(url, output, quiet=False)

print("All models downloaded successfully!")