import argparse
import warnings
warnings.filterwarnings("ignore")

from os.path import join, exists
from os import makedirs
from time import localtime, strftime

time = strftime("%y_%m_%d-%H_%M_%S", localtime())
parser = argparse.ArgumentParser(description="CQA System")

####################################
#       Environment Setting        #
####################################
parser.add_argument('-d', '--dataset', default='dataset', type=str,
                    help="The directory path where the dataset is stored")
parser.add_argument('--root_dir', default='.', type=str,
                    help="The directory path where all the computed data is stored")
parser.add_argument('-r', '--restore_model', action='store_true',
                    help="Restore a model from previous version")
parser.add_argument('-w', '--weights', default='model_19_01_25-12_05_15.h5', type=str,
                    help="Load specific weights to the model")
parser.add_argument('--force_update', action='store_true', help="Forcefully recalculate entire project")
parser.add_argument('--debug', action='store_true', help="Debug the program")

####################################
#     Hyper Parameter Setting      #
####################################


args = parser.parse_args()


args.time = time
args.logged_dir = join(args.root_dir, args.time)

if not exists(args.logged_dir):
    # makedirs(args.logged_dir)
    pass

if args.debug:
    print("=" * 53)
    print("ARGUMENTS".ljust(25) + "  " + "VALUES")
    print("=" * 25 + "  " + "=" * 25)
    for arg in vars(args):
        print(str(arg).ljust(25) + " |" + str(getattr(args, arg)))
    print("=" * 53)