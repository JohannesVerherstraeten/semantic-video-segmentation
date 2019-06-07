"""
This file configures the datasets. This allows running the code on multiple
computers, independent from the directory structure.

Update the base directories to match the setup of your project.

To avoid pushing your personal config file to git repo, do:
git update-index --assume-unchanged config.py

"""

# =======================
# DIRECTORY CONFIGURATION
# =======================

# ----------------
# Base directories
# ----------------


PROJECT_DIR = "/media/johannes/Data/documenten-van-johannes/Documenten/School/KU Leuven/Master 2/Thesis/temporal/"
CAMVID_DIR = "/media/johannes/Seagate Backup Plus Drive/Datasets/CamVid/"
CITYSCAPES_DIR = "/media/johannes/Seagate Backup Plus Drive/Datasets/Cityscapes/"


# ----------------
# Dataset directories
# -------------------

# CamVid

# CamVid train-, validation- and test directories
CAMVID_TRAIN_DIR = CAMVID_DIR + 'train'
CAMVID_TRAIN_LABEL_DIR = CAMVID_DIR + 'trainannot'

CAMVID_VAL_DIR = CAMVID_DIR + 'val'
CAMVID_VAL_LABEL_DIR = CAMVID_DIR + 'valannot'

CAMVID_TEST_DIR = CAMVID_DIR + 'test'
CAMVID_TEST_LABEL_DIR = CAMVID_DIR + 'testannot'

# CamVid filename filters. Only filenames satisfying this filter are loaded.
CAMVID_FILE_FILTER = lambda x: x.endswith(".png")
CAMVID_VIDEO_FILE_FILTER = lambda x: x.endswith(".avi") or x.endswith(".MXF")
CAMVID_LABEL_FILE_FILTER = lambda x: x.endswith(".png")


# Cityscapes

# Cityscapes train-, validation- and test directories
CITYSCAPES_TRAIN_DIR = CITYSCAPES_DIR + "leftImg8bit/train"
CITYSCAPES_TRAIN_LABEL_DIR = CITYSCAPES_DIR + "gtFine/train"
CITYSCAPES_TRAIN_SEQ_DIR = CITYSCAPES_DIR + "leftImg8bit_sequence/train"

CITYSCAPES_VAL_DIR = CITYSCAPES_DIR + "leftImg8bit/val"
CITYSCAPES_VAL_LABEL_DIR = CITYSCAPES_DIR + "gtFine/val"
CITYSCAPES_VAL_SEQ_DIR = CITYSCAPES_DIR + "leftImg8bit_sequence/val"

CITYSCAPES_TEST_DIR = CITYSCAPES_DIR + "leftImg8bit/test"
CITYSCAPES_TEST_LABEL_DIR = CITYSCAPES_DIR + "gtFine/test"
CITYSCAPES_TEST_SEQ_DIR = CITYSCAPES_DIR + "leftImg8bit_sequence/test"

# Cityscapes filename filters. Only filenames satisfying this filter are loaded.
CITYSCAPES_FILE_FILTER = lambda x: x.endswith(".png")
CITYSCAPES_LABEL_FILE_FILTER = lambda x: x.endswith("labelIds.png")
