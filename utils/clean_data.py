import logging as logger
import os
import shutil
import tarfile
from csv import reader

import cv2
import numpy as np
import pandas as pd

##################
# IMAGE CLEANING #
##################


def ravel_images(parent_dir: str, final_dir: str):
    photos = [p for p in os.listdir(parent_dir) if p.endswith(".png")]
    lines = []

    for photo in photos:
        img = cv2.cvtColor(
            cv2.imread(os.path.join(parent_dir, photo)),
            cv2.COLOR_BGR2GRAY,
        )
        ravelled = img.ravel()
        lines.append(ravelled)

    with open(os.path.join(final_dir, "ravelled_image_data.csv"), "w") as file:
        for line in lines:
            file.write(",".join(str(x) for x in line))
            file.write("\n")


def extract_images(
    parent_dir: str,
    target_dir: str,
):

    tar_files = [f for f in os.listdir(parent_dir) if f.endswith(".tar.bz2")]

    for tar_file in tar_files:
        tar_path = os.path.join(parent_dir, tar_file)
        tar = tarfile.open(tar_path, "r:bz2")
        tar.extractall(os.path.join(target_dir, "extraction"))
        tar.close()

    for folder in os.listdir(os.path.join(target_dir, "extraction")):
        for file in os.listdir(os.path.join(target_dir, "extraction", folder)):
            shutil.move(
                os.path.join(target_dir, "extraction", folder, file),
                os.path.join(target_dir, file),
            )


#################
# TEXT CLEANING #
#################


def add_trial_column(
    parent_dir: str,
    file: str,
):
    # use the name of the file to get the trial id
    # add trial identifier to df before combining
    splits = file.split("_")
    iden = "_".join(splits[0:2])
    file_df = pd.read_csv(os.path.join(parent_dir, file))
    file_df["trial_id"] = iden
    return file_df


def combine_clean_files(
    parent_dir: str,
    final_dir: str,
):
    files = [f for f in os.listdir(parent_dir) if f.endswith(".txt")]
    combined_df = pd.concat([add_trial_column(parent_dir, f) for f in files])
    dest_path = os.path.join(final_dir, "combined_trial_data.csv")
    combined_df.to_csv(dest_path, index=False)


def clean_ms_pacman(df):
    # episode id isn't applicable for ms pacman, always 0
    df.drop("episode_id", axis=1, inplace=True)
    return df


def add_action_strings(df):
    # make mapping for actions and add a column for the string mappings
    # rename the current action column to action_int
    mapping = {
        "0": "PLAYER_A_NOOP",
        "1": "PLAYER_A_FIRE",
        "2": "PLAYER_A_UP",
        "3": "PLAYER_A_RIGHT",
        "4": "PLAYER_A_LEFT",
        "5": "PLAYER_A_DOWN",
        "6": "PLAYER_A_UPRIGHT",
        "7": "PLAYER_A_UPLEFT",
        "8": "PLAYER_A_DOWNRIGHT",
        "9": "PLAYER_A_DOWNLEFT",
        "10": "PLAYER_A_UPFIRE",
        "11": "PLAYER_A_RIGHTFIRE",
        "12": "PLAYER_A_LEFTFIRE",
        "13": "PLAYER_A_DOWNFIRE",
        "14": "PLAYER_A_UPRIGHTFIRE",
        "15": "PLAYER_A_UPLEFTFIRE",
        "16": "PLAYER_A_DOWNRIGHTFIRE",
        "17": "PLAYER_A_DOWNLEFTFIRE",
    }
    df.rename(columns={"action": "action_int"}, inplace=True)
    df["action_str"] = df.apply(lambda x: mapping[x["action_int"]], axis=1)

    return df


def clean_single_text_file(
    filepath: str,
    target_path: str,
):

    frame_ids = []
    episode_ids = []
    scores = []
    durations = []
    unclipped_rewards = []
    actions = []
    gaze_positions_x = []
    gaze_positions_y = []

    with open(filepath) as obj:
        csv_reader = reader(obj)
        _ = next(csv_reader)  # throw away the header
        for row in csv_reader:
            # Starting with the 6th entry in each row are
            # a variable number of x and y gaze positions
            # Turn these entries long so each row has only
            # one set of x and y gaze positions
            start = 6
            end = len(row) - 1
            for i in range(start, end, 2):
                current_x = row[i]
                current_y = row[i + 1]
                frame_ids.append(row[0])
                episode_ids.append(row[1])
                scores.append(row[2])
                durations.append(row[3])
                unclipped_rewards.append(row[4])
                actions.append(row[5])
                gaze_positions_x.append(current_x)
                gaze_positions_y.append(current_y)

    df = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "episode_id": episode_ids,
            "score": scores,
            "duration": durations,
            "unclipped_reward": unclipped_rewards,
            "action": actions,
            "gaze_position_x": gaze_positions_x,
            "gaze_position_y": gaze_positions_y,
        }
    )

    try:
        df = add_action_strings(df)
    except Exception as ex:
        logger.error(
            f"encountered exception trying add_action_strings() for file {filepath}"
        )
        logger.error(f"exception: {ex}")

    # below is for any game-specific cleaning
    # altering or adding cleaning for games besides ms pacman
    #  could have downstream effects
    if "ms_pacman" in filepath:
        try:
            df = clean_ms_pacman(df)
        except Exception as ex:
            logger.error(
                f"encountered exception trying clean_ms_pacman() for file {filepath}"
            )
            logger.error(f"exception: {ex}")

    # write cleaned data frame to target_dir
    df.to_csv(target_path, index=False)


def clean_each_text_file(
    parent_dir: str,
    target_dir: str,
):
    # get all text files present in the directory
    files = [f for f in os.listdir(parent_dir) if f.endswith(".txt")]

    # for each file, clean file
    for file in files:
        filepath = os.path.join(parent_dir, file)
        target_path = os.path.join(target_dir, file)
        try:
            clean_single_text_file(filepath, target_path)
        except Exception as ex:
            logger.error(
                f"encountered exception trying clean_single_text_file() for file {file}"
            )
            logger.error(f"exception: {ex}")


def ensure_directory(dir_name: str):
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        pass


###########################
# THE MOTHERSHIP FUNCTION #
###########################


def clean_all_raw_data(
    game_name: str,
    highscore: bool,
    source_dir: str = "raw_data",
    target_dir: str = "cleaned_data",
    final_dir: str = "final_data",
):

    # check file paths
    # for our project:
    #  -- cleaning_parent_dir = raw_data/ms_pacman/highscore/
    #  -- cleaning_target_dir = cleaned_data/ms_pacman/highscore/
    #  -- combining_parent_dir = cleaned_data/ms_pacman/highscore/
    #  -- final_target_dir = final_data/ms_pacman/highscore/
    cleaning_parent_dir = os.path.join(source_dir, game_name)
    cleaning_target_dir = os.path.join(target_dir, game_name)
    final_target_dir = os.path.join(final_dir, game_name)
    if highscore:
        cleaning_parent_dir = os.path.join(cleaning_parent_dir, "highscore")
        cleaning_target_dir = os.path.join(cleaning_target_dir, "highscore")
        final_target_dir = os.path.join(final_target_dir, "highscore")
    combining_parent_dir = cleaning_target_dir
    final_target_dir_images = os.path.join(final_target_dir, "frames")

    # if the source_dir with the raw data doesn't exist, raise error
    # any others, ensure they exist
    if not os.path.isdir(cleaning_parent_dir):
        raise FileNotFoundError()
    ensure_directory(combining_parent_dir)
    ensure_directory(final_target_dir)
    ensure_directory(cleaning_target_dir)
    ensure_directory(final_target_dir_images)

    # clean each text file
    try:
        clean_each_text_file(
            parent_dir=cleaning_parent_dir,
            target_dir=cleaning_target_dir,
        )
    except Exception as ex:
        logger.error("encountered exception trying to execute clean_each_text_file()")
        logger.error(f"exception: {ex}")

    # combine all cleaned text files
    try:
        combine_clean_files(
            parent_dir=combining_parent_dir,
            final_dir=final_target_dir,
        )
    except Exception as ex:
        logger.error("encountered exception trying to execute combine_clean_files()")
        logger.error(f"exception: {ex}")

    # extract images
    try:
        extract_images(
            parent_dir=cleaning_parent_dir,
            target_dir=final_target_dir_images,
        )
    except Exception as ex:
        logger.error("encountered exception trying to execute extract_images()")
        logger.error(f"exception: {ex}")

    # ravel images
    try:
        ravel_images(
            parent_dir=final_target_dir_images,
            final_dir=final_target_dir,
        )
    except Exception as ex:
        logger.error("encountered exception trying to execute ravel_images()")
        logger.error(f"exception: {ex}")


def __main__():

    # default parameters
    GAME_NAME = "ms_pacman"
    HIGHSCORE = True
    SOURCE_DIR = "raw_data"
    TARGET_DIR = "cleaned_data"
    FINAL_DIR = "final_data"

    clean_all_raw_data(
        game_name=GAME_NAME,
        highscore=HIGHSCORE,
        source_dir=SOURCE_DIR,
        target_dir=TARGET_DIR,
        final_dir=FINAL_DIR,
    )
