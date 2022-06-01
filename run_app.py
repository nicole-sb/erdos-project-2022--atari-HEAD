import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Eye Gaze Data with Ms Pacman",
    page_icon=":joystick:",
    layout="wide",
)

####################
# DEFINE FUNCTIONS #
####################


@st.cache
def get_random_photo():
    dir = "final_data/ms_pacman/highscore/frames"
    photos = os.listdir(dir)
    max = len(photos) - 1
    rand_int = np.random.randint(low=0, high=max)
    return os.path.join(dir, photos[rand_int])


def get_dummy_model_response():
    return np.random.randint(low=1, high=4)


##########################
# DRAWABLE CANVAS CONFIG #
##########################

# Specify canvas parameters in application
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

st.markdown("# Eye Gaze Data with Ms Pacman")
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
) = st.columns(2)

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
    gaze_x = (objects["left"][0]) / 2
    gaze_y = (420 - objects["top"][0]) / 2

with col2:
    st.markdown(gaze_x.astype("int"))
    st.markdown(gaze_y.astype("int"))

    st.markdown(get_dummy_model_response())
