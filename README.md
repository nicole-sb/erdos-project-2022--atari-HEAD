# Atari-HEAD: Game images and decision-making prediction
## *Nicole Basinski, Nydia Chang, Danny Wan, Jason Xing*

## Project Intro

As part of the Erdos May Bootcamp 2022, we submit our project using gaze position data on Atari videogames, obtained from the [Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset](https://zenodo.org/record/3451402#.YpEEB5PML0r).

20 Atari games were played by 4 human players in a frame-by-frame manner to obtain gaze samples, as well as associated action taken by the player, current game score, among some other information. Semi-frame-by-frame gameplay allowed for the players to make near-optimal game decisions that led to scores in the range of known human records. Semi-frame-by-frame gameplay also resulted in more accurate game state and action associations (due to removing the effect of human reaction time); this resulted in more optimal data for any supervised learning algorithms.

Regular trials were set to a 15-minute timeframe; highscore trials allowed the player to continue gameplay until they ran out of lives (up to a max of 2 hours). Each trial, whether regular or highscore, corresponds to a text file and a .tar.bz2 file. The text file recorded player actions, gaze positions, and other data for each frame during that trial, and the .tar.bz2 file includes .png images of each game frame.

This project will use the game frame images along with associated gaze positions to model the best resulting action. We focused on the Ms. Pacman game and specifically the `highscore` trials, largely for time and computational reasons. However, much of the modeling included in this repo would apply for other games and trials as well.

## Project Goals & Applications
**Problem:** It is a difficult task to get information on players decision making process in video games. Most information on this subject is limited to players’ inputs into their controllers and speculations. One can only make an educated guess on what players were thinking about before making inputs.

**Solution:** Team Apollo created a model that utilizes a player’s gaze positions and information from video game frames to predict their actions before they are made. This information on quantitative data on players’ decision making process.

**Stakeholders:** Our stakeholders can be divided into two categories. A primary stakeholder that is directly affected by our model are game developers. Secondary stakeholders that would have an interest in our model are hardware and software companies for eye-trackers and virtual reality, and gamers.

# Modeling Method
A multi-layer perceptron (MLP) was trained using 80% of data extracted from each frame during a high-score 2-hour playthrough of Ms. Pac-Man from the Atari Human Eye-Tracking and Demonstration dataset. The trained MLP was then tested with the remaining 20% of the data. The model yielded an accuracy of 73% on the test data. This process can be found in the `Data_prediction` folder of this repository.

# In this repository
The `Data_handling` folder includes a notebook that walks through some feature extraction for the game play images we use in this project, as well as some supplementary files for this notebook.

The `Data_prediction` folder includes a notebook that walks through our modeling process (explained above in the Modeling Method section), as well as some supplementary files for this notebook.

`final_data/ms_pacman/highscore/frames` includes all the game play images used in this project, sourced from the original dataset (Zhang et al.).

The code for our Streamlit app can be found under `run_app.py.` Some utilities the app uses are found under `utils/`. Unfortunately, the deployed app does not function due to an [unresolved import issue](https://discuss.streamlit.io/t/cannot-see-streamlit-drawable-canvas/6235) with a custom package used in the app. The app can be run locally if the user clones the repo, installs the packages found in requirements.txt, and runs `streamlit run run_app.py'.


# Appendix
**Citations:**

Ruohan Zhang, Calen Walshe, Zhuode Liu, Lin Guan, Karl S. Muller, Jake A. Whritner, Luxin Zhang, Mary Hayhoe, & Dana Ballard. (2019). Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset (Version 4) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3451402
