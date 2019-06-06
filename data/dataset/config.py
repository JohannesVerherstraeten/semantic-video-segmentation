"""
This file configures ENet and the datasets. This allows running the code on multiple
computers, independent from the directory structure.

To avoid pushing your personal config file, do:
git update-index --assume-unchanged ENet-modified/.config.py

"""

# =======================
# DIRECTORY CONFIGURATION
# =======================

# ----------------
# Base directories
# ----------------

# Johannes personal pc
PROJECT_DIR = "/media/johannes/Data/documenten-van-johannes/Documenten/School/KU Leuven/Master 2/Thesis/temporal/"
CAMVID_DIR = "/media/johannes/Seagate Backup Plus Drive/Datasets/CamVid/"
CITYSCAPES_DIR = "/media/johannes/Seagate Backup Plus Drive/Datasets/Cityscapes/"

# ESAT pc
# PROJECT_DIR = "/users/start2016/r0582208/Documents/temporal/ENet-modified/"
# CAMVID_DIR = "/esat/asahi/r0582208/CamVid/"
# CITYSCAPES_DIR = "/esat/asahi/r0582208/Cityscapes/"


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
