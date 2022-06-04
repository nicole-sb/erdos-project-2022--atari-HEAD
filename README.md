# Atari-HEAD: Game images and decision-making prediction
## *Nicole Basinski, Nydia Chang, Danny Wan, Jason Xing*

## Project Intro

As part of the Erdos May Bootcamp 2022, we submit our project using gaze position data on Atari videogames, obtained from the [Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset](https://zenodo.org/record/3451402#.YpEEB5PML0r).

20 Atari games were played by 4 human players in a frame-by-frame manner to obtain gaze samples, as well as associated action taken by the player, current game score, among some other information. Semi-frame-by-frame gameplay allowed for the players to make near-optimal game decisions that led to scores in the range of known human records. Semi-frame-by-frame gameplay also resulted in more accurate game state and action associations (due to removing the effect of human reaction time); this resulted in more optimal data for any supervised learning algorithms.

Regular trials were set to a 15-minute timeframe; highscore trials allowed the player to continue gameplay until they ran out of lives (up to a max of 2 hours). Each trial, whether regular or highscore, corresponds to a text file and a .tar.bz2 file. The text file recorded player actions, gaze positions, and other data for each frame during that trial, and the .tar.bz2 file includes .png images of each game frame.

This project will use the game frame images along with associated gaze positions to model the best resulting action. We focused on the Ms. Pacman game and specifically the `highscore` trials, largely for time and computational reasons. However, much of the modeling included in this repo would apply for other games and trials as well.

## Project Goals & Applications
**Problem:** It is difficult to design video games that are adaptable to players. Artificial intelligence (AI) in games are set to have specific behaviors using algorithms that are predictable by players. This leads to decreases in player engagement because players can anticipate in-game events.

**Solution:** Team Apollo created a model that utilizes a player’s gaze positions and information from video game frames to predict their actions before they are made. This information on human decision-making can be used to create a more adaptable and dynamic video game experience to increase player engagement.

**Stakeholders:** Our stakeholders can be divided into two categories. A primary stakeholder that is directly affected by our model are game developers. Secondary stakeholders that would have an interest in our model are hardware and software companies for eye-trackers and virtual reality, and gamers.
