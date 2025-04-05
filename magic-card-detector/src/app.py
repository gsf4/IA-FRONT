from magic_card_detector.model.detector import MagicCardDetector
import streamlit as st
from magic_card_detector.utils.image_processing import preprocess_image, display_image

def main():
    st.title("Magic Card Detector")
    st.write("Upload a JPEG image to check if it contains a Magic card.")

    uploaded_file = st.file_uploader("Choose a JPEG image...", type="jpeg")

    if uploaded_file is not None:
        # Preprocess the image
        image = preprocess_image(uploaded_file)
        display_image(image)

        # Load the model and make a prediction
        detector = MagicCardDetector()
        detector.load_model()
        result = detector.predict(image)

        if result:
            st.success("This image contains a Magic card!")
        else:
            st.error("No Magic card detected in this image.")

if __name__ == "__main__":
    main()