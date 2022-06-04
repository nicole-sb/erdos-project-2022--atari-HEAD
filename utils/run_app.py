import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from model import get_model_response, get_photo_information, train_model

st.set_page_config(
    page_title="Atari-HEAD",
    page_icon=":joystick:",
    layout="wide",
)

####################
# DEFINE FUNCTIONS #
####################


@st.cache
def get_random_photo():
    dir = "../final_data/ms_pacman/highscore/frames"
    photos = os.listdir(dir)
    max = len(photos) - 1
    rand_int = np.random.randint(low=0, high=max)
    return os.path.join(dir, photos[rand_int])


@st.cache(allow_output_mutation=True)
def call_train_model():
    model = train_model()
    return model


##########################
# DRAWABLE CANVAS CONFIG #
##########################

# Specify canvas parameters in application
model = call_train_model()
photo = get_random_photo()
drawing_mode = "point"
stroke_width = 3
point_display_radius = 3
stroke_color = "yellow"
bg_color = "gray"
bg_image = Image.open(photo)
realtime_update = True


#############
# PAGE TEXT #
#############

st.markdown("# Atari-HEAD: Game images and decision-making prediction")
st.markdown("### Nicole Basinski, Nydia Chang, Danny Wan, & Jason Xing")
st.markdown("")
st.markdown(
    """
    Our model uses a single frame of a Ms Pacman game, as well as eye gaze data, to predict the next move a human might choose in an optimal setting.

    To simulate eye gaze data, please click on the image where you would like the 'eye gaze' target to be. *Hint: In gameplay, the human eye gaze tends to be where the player is planning to go next.*
    """
)
st.markdown("")
st.markdown("")


##############
# PAGE SETUP #
##############

(
    col1,
    col2,
) = st.columns([1, 2])

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=bg_image,
        update_streamlit=realtime_update,
        height=420,
        width=320,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        key="canvas",
    )

if canvas_result.json_data is not None:
    objects = pd.json_normalize(
        canvas_result.json_data["objects"]
    )  # need to convert obj to str because PyArrow
    if not objects.empty:
        gaze_x = (objects["left"][0]) / 2
        gaze_y = (420 - objects["top"][0]) / 2

with col2:
    if not objects.empty:

        X = get_photo_information(photo, gaze_x.astype("int"), gaze_y.astype("int"))

        action = get_model_response(model, X)

        st.markdown("Predicted next action: ")
        st.markdown("#### " + action)

        st.markdown("")
        st.markdown("")

        st.markdown(
            "Gaze Position *in pixels where (0,0) is the bottom left corner and (160,210) is the top right corner of the image*"
        )
        st.markdown("#### (" + gaze_x.astype("str") + "," + gaze_y.astype("str") + ")")
