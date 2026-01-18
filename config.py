W_TARGET = 640
DEBUG = False
BASE_PATH = <FILL_ME>
MATCH_METHOD="brute_force"  # flann_ratio_test, brute_force_ratio_test, brute_force

# Max distance (px) from point to epipolar line to be considered an inlier (typically 1-3).
RANSAC_THRESHOLD = 0.1
RANSAC_PROB = 0.9999

RANSAC_MAX_ITERS = 2000
KITTY_SEQUENCE = "00"

DETECTOR = "sift"  # rootsift, asift, sift