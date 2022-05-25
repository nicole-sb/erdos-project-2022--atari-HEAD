import logging as logger
import os
import shutil
import tarfile
from csv import reader

import cv2
import numpy as np
import pandas as pd

# this script pulls in all text files in the given source_dir
#  and performs some data cleaning. Then writes the new files
#  to the given target_dir. The clean trial files are then
#  combined into a single dataframe and written to the same
#  target_dir as `combined.csv`


################
# COMMON UTILS #
################


def ensure_directory(dir_name):
    if os.path.exists(dir_name):
        pass
    else:
        os.mkdir(dir_name)


###########################
# DATA TEXT FILE CLEANING #
###########################


def add_trial_column(filepath, game):
    # use the name of the file to get the trial id
    # add trial identifier to df before combining
    file_df = pd.read_csv(os.path.join("cleaned_data", game, filepath))
    splits = filepath.split("_")
    iden = "_".join(splits[0:3])
    file_df["trial_id"] = iden
    return file_df


def combine_trial_files(game):
    files = [
        f for f in os.listdir(os.path.join("cleaned_data", game)) if f.endswith(".txt")
    ]
    full_df = pd.concat([add_trial_column(f, game) for f in files])
    dest_path = os.path.join("cleaned_data", game, "combined.csv")
    full_df.to_csv(dest_path, index=False)
    logger.info(f"combined dataframe written to {dest_path}")
    return dest_path


def clean_ms_pacman(df):
    # episode id isn't applicable for ms pacman, always 0
    return df.drop("episode_id", axis=1, inplace=True)


def clean_all_games(df):
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


def clean_data_file(fp, target_dir, game):
    try:
        frame_ids = []
        episode_ids = []
        scores = []
        durations = []
        unclipped_rewards = []
        actions = []
        gaze_positions_x = []
        gaze_positions_y = []

        with open(fp) as obj:
            csv_reader = reader(obj)
            _ = next(csv_reader)
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

        df = clean_all_games(df)

        try:
            head, tail = os.path.split(fp)
            _, head_tail = os.path.split(head)
            dir_folder = os.path.join(target_dir, head_tail)

            if game == "ms_pacman":
                clean_ms_pacman(df)

            ensure_directory(dir_folder)

            filepath = os.path.join(dir_folder, tail)
            df.to_csv(filepath, index=False)
            logger.info(f"cleaned data written to {filepath}")

        except Exception as ex:
            logger.error(f"error while writing data for {fp}")
            logger.error(f"error: {ex}")

    except Exception as ex:
        logger.error(f"error while creating dataframe for {fp}")
        logger.error(f"error: {ex}")


def clean_all_raw_data(source_dir, target_dir, game):
    dirs = [
        os.path.join(source_dir, d)
        for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    for one_dir in dirs:
        files = [f for f in os.listdir(one_dir) if f.endswith(".txt")]
        for one_file in files:
            filepath = os.path.join(one_dir, one_file)
            clean_data_file(filepath, target_dir, game)

    combined_fp = combine_trial_files(game)

    return combined_fp


############################
# DATA IMAGE FILE CLEANING #
############################


def extract_images(source_dir, target_dir, game):
    dirs = [
        os.path.join(source_dir, d)
        for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    for one_dir in dirs:
        tar_files = [f for f in os.listdir(one_dir) if f.endswith(".tar.bz2")]

        for tar_tail in tar_files:
            tar_path = os.path.join(one_dir, tar_tail)
            tar = tarfile.open(tar_path, "r:bz2")

            ensure_directory("extraction")
            ensure_directory(os.path.join(target_dir, game, "all_images"))

            tar.extractall(os.path.join("extraction"))
            tar.close()

        for folder in os.listdir("extraction"):
            for file in os.listdir(os.path.join("extraction", folder)):
                shutil.move(
                    os.path.join("extraction", folder, file),
                    os.path.join(target_dir, game, "all_images"),
                )

    return os.path.join(target_dir, game, "all_images")


def ravel_images(image_dir):
    photos = [p for p in os.listdir(image_dir) if p.endswith(".png")]
    lines = []

    for photo in photos:
        img = cv2.cvtColor(
            cv2.imread(os.path.join(image_dir, photo)),
            cv2.COLOR_BGR2RGB,
        )
        ravelled = img.ravel()
        lines.append(ravelled)

    with open(os.path.join(image_dir, "ravelled_image_data.csv"), "w") as file:
        for line in lines:
            file.write(np.array2string(line, separator=","))
            file.write("\n")

    return os.path.join(image_dir, "ravelled_image_data.csv")
