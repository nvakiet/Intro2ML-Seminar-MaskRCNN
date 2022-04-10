import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as pyplot
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.columns(2)

    col1.subheader("Original Image")
    st.text("")
    pyplot.figure(figsize = (15,15))
    pyplot.imshow(our_image)
    col1.pyplot(use_column_width=True)
    st.text("")
    col2.subheader("Object-Detected Image")
    st.text("")

    # define classes
    class_names = ['BG', 'person', 'car', 'bus']
 
    # define the test configuration
    class TestConfig(Config):
        NAME = "test"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80
 
    # define the model
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
    # load model weights
    rcnn.load_weights('model.h5', by_name=True)
    # load photograph
    img = our_image
    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=0)
    # get dictionary for first prediction
    r = results[0]
    # show photo with bounding boxes, masks, class labels and scores
    display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    col2.pyplot(use_column_width=True)

def object_main():
    """OBJECT DETECTION APP"""

    st.title("Group05 - Mask-RCNN demo")
    st.write("Mask-RCNN demo for person, car and bus detection.")

    choice = st.radio("", ("Browse an Image", "Show Demo"))
    st.write()

    if choice == "Browse an Image":
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload Image", type=['jpg','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            detect_objects(our_image)

    elif choice == "Show Demo":
        our_image = Image.open("demo.jpg")
        detect_objects(our_image)

if __name__ == '__main__':
    object_main()