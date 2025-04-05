def preprocess_image(image):
    from PIL import Image
    import numpy as np

    # Convert the image to RGB format
    image = image.convert("RGB")
    
    # Resize the image to the required input size for the model
    image = image.resize((224, 224))  # Example size, adjust as necessary
    
    # Convert the image to a numpy array and normalize
    image_array = np.array(image) / 255.0
    
    return image_array

def display_image(image):
    import matplotlib.pyplot as plt

    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()