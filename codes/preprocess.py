from PIL import Image
import os
folder=[]

path='C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\train'
for filename in os.listdir(path):
  if not (filename.endswith("_forg")):
    folder.append(os.path.join(path, filename))
desired_size = (224,224)

# Create a new folder to store the resized images
resized_folder_path = 'C:\\Users\\pshab\\Desktop\\signature verification\\signver\\dataset\\Real'
if not os.path.exists(resized_folder_path):
    os.makedirs(resized_folder_path)

for folder_path in folder:
  for filename in os.listdir(folder_path):
      if filename.endswith(".PNG") or filename.endswith(".png"):
          # Open the image file
          img_path = os.path.join(folder_path, filename)
          original_img = Image.open(img_path)
        # Resize the image
          resized_img = original_img.resize(desired_size)
        # Save the resized image to the new folder
          resized_img_path = os.path.join(resized_folder_path, filename)
          resized_img.save(resized_img_path)



folder2=[]
path='C:\\Users\\pshab\\Desktop\\signature verification\\signver\\archive\\sign_data\\train'
for filename in os.listdir(path):
  if (filename.endswith("_forg")):
    folder.append(os.path.join(path, filename))
desired_size = (224, 224)

# Create a new folder to store the resized images
resized_folder_path = 'C:\\Users\\pshab\\Desktop\\signature verification\\signver\\dataset\\Fake'
if not os.path.exists(resized_folder_path):
    os.makedirs(resized_folder_path)
for folder_path in folder:
# Loop through all image files in the folder
  for filename in os.listdir(folder_path):
      if filename.endswith(".PNG") or filename.endswith(".png"):
          # Open the image file
          img_path = os.path.join(folder_path, filename)
          original_img = Image.open(img_path)
        # Resize the image
          resized_img = original_img.resize(desired_size)
        # Save the resized image to the new folder
          resized_img_path = os.path.join(resized_folder_path, filename)
          resized_img.save(resized_img_path)







full_dataset1 = tf.data.Dataset.from_generator(
    lambda: dataset_generator_function(training_csv=testing_csv, training_dir=testing_dir, transform=transform),
    output_signature=(
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),  # Scalar label
    )
)






concatenated = tf.concat([, x1], axis=1)
    concatenated = concatenated.numpy().squeeze()
    plt.imshow(concatenated, cmap='gray')
    plt.title(f'Dissimilarity: {euclidean_distance.numpy()[0]:.2f} Label: {label_text}')
    plt.show()
    # Clean up the temporarily saved files
    os.remove(original_signature_path)
    os.remove(uploaded_file_path)












from PIL import Image
from train import model_predict, extraction
import numpy as np

def load_model_and_verify(uploaded_image_path):
    uploaded_image = preprocess_image(uploaded_image_path)
    features_batch = extraction(uploaded_image)

    time_steps = 10
    num_features_per_time_step = 3584 // time_steps
    train_features = features_batch.reshape(1, time_steps, num_features_per_time_step)
    
    prediction = model_predict(train_features)
    result = "Real" if prediction == 0 else "Fake"
    return result

def preprocess_image(image_path):
    original_img = Image.open(image_path)
    image = original_img.resize((320, 240))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array













setOriginalSignature(response.data.org)
        setUploadedSignature(response.data.uploaded)

<div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <img src={originalSignature} alt="Original Signature" style={{ width: '10%' }} />
                  <img src={uploadedSignature} alt="Uploaded Signature" style={{ width: '10%' }} />
                </div>

,'org':original_signature_path,'uploaded':uploaded_file_path