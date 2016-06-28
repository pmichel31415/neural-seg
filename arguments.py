import argparse

parser = argparse.ArgumentParser(
    description='Neural speech segmentation'
)

###############################
# General execution arguments #
###############################

parser.add_argument(
    '-N', '--name',
    action='store',
    dest='name',
    default=None,
    type=str,
    help='Name of the experiment'
)
parser.add_argument(
    '-w', '--workflow',
    action='store',
    dest='workflow',
    default='train|test|post_process|eval',
    type=str,
    help='Tasks that need to be run (any subset '
    'of mfcc|preprocess|train|test|postprocess|eval)'
)
parser.add_argument(
    '-v', '--verbose',
    help='Increase output verbosity',
    action='store_true'
)
parser.add_argument(
    '-J', '--json',
    action='store',
    dest='json',
    default=None,
    type=str,
    help='Json file containing parameters'
)

######################################
# Various directories and file lists #
######################################

parser.add_argument(
    '-train', '--train_data',
    action='store',
    dest='train_list',
    default=None,
    type=str,
    help='File listing training files locations'
)
parser.add_argument(
    '-test', '--test_data',
    action='store',
    dest='test_list',
    default=None,
    type=str,
    help='File listing test files locations'
)
parser.add_argument(
    '-wav', '--wav_dir',
    action='store',
    dest='wav_dir',
    default=None,
    type=str,
    help='Directory containing wav files'
)
parser.add_argument(
    '-mfcc', '--mfcc_dir',
    action='store',
    dest='mfcc_dir',
    default=None,
    type=str,
    help='Location of mfcc files'
)
parser.add_argument(
    '-states', '--states_dir',
    action='store',
    dest='states_dir',
    default=None,
    type=str,
    help='Location of states files'
)
parser.add_argument(
    '-t', '--times',
    action='store',
    dest='time_dir',
    default=None,
    type=str,
    help='Directory containing times files'
)
parser.add_argument(
    '-gold', '--gold_dir',
    action='store',
    dest='gold_dir',
    default=None,
    type=str,
    help='Directory containing gold boundaries files'
)
parser.add_argument(
    '-o', '--out_dir',
    action='store',
    dest='out_dir',
    default=None,
    type=str,
    help='Output dir'
)

#############################
# Neural networks arguments #
#############################

parser.add_argument(
    '-m', '--model',
    action='store',
    dest='model',
    default=None,
    type=str,
    help='Previously saved model'
)
parser.add_argument(
    '-de', '--dim_embed',
    action='store',
    dest='embed_dim',
    default=39,
    type=int,
    help='Embedding dimension'
)
parser.add_argument(
    '-dh', '--dim_hidden',
    action='store',
    dest='hidden_dim',
    default=50,
    type=int,
    help='Hidden layer dimension'
)
parser.add_argument(
    '-s', '--span',
    action='store',
    dest='span',
    default=7,
    type=int,
    help='BPTT time span'
)
parser.add_argument(
    '-bs', '--batch_size',
    action='store',
    dest='batch_size',
    default=10,
    type=int,
    help='Batch size'
)
parser.add_argument(
    '-optim', '--optimizer',
    action='store',
    dest='optim',
    default='sgd',
    type=str,
    help='Optimizer'
)
parser.add_argument(
    '-loss', '--loss_function',
    action='store',
    dest='loss',
    default='mse',
    type=str,
    help='Loss function'
)
parser.add_argument(
    '-l', '--lr',
    action='store',
    dest='lr',
    default=0.01,
    type=float,
    help='Learning rate'
)
parser.add_argument(
    '-trmt', '--train_model_type',
    action='store',
    dest='train_model_type',
    default='simple_rnn',
    type=str,
    help='Model type to be used for training'
)
parser.add_argument(
    '-temt', '--test_model_type',
    action='store',
    dest='test_model_type',
    default='simple_rnn',
    type=str,
    help='Model type to be used for testing'
)
parser.add_argument(
    '-st', '--stateful',
    help='Use stateful training',
    action='store_true'
)

##############################
# MFCC computation arguments #
##############################

parser.add_argument(
    '-me', '--mfcc_energy',
    help='Include nergy in mfcc',
    action='store_true'
)
parser.add_argument(
    '-nc', '--numcep',
    action='store',
    dest='numcep',
    default=13,
    type=int,
    help='Number of cepstral coeffs in mfcc'
)

###########################
# Preprocessing arguments #
###########################

parser.add_argument(
    '-Q', '--cluster_num',
    action='store',
    dest='num_clusters',
    default=8,
    type=int,
    help='Number of clusters for preprocessing'
)
parser.add_argument(
    '-S', '--subset_size',
    action='store',
    dest='preprocess_subset_size',
    default=10000,
    type=int,
    help='Preprocessing subset size (for kmeans)'
)
parser.add_argument(
    '-prepm', '--preprocess_method',
    action='store',
    dest='preprocess_method',
    default='random',
    choices=['random', 'kmeans', 'partial_kmeans'],
    type=str,
    help='Method to be used for pre processing'
)

#############################
# Post processing arguments #
#############################

parser.add_argument(
    '-postpm', '--postprocess_method',
    action='store',
    dest='postprocess_method',
    default='baseline',
    choices=['manual', 'auto', 'baseline', 'fourier', 'greedy'],
    type=str,
    help='Method to be used for post processing'
)
parser.add_argument(
    '-thr', '--threshold',
    action='store',
    dest='threshold',
    default=0.5,
    type=float,
    help='Peak detection threshold'
)
parser.add_argument(
    '-K', '--kernel_len',
    action='store',
    dest='ker_len',
    default=3.0,
    type=float,
    help='Convolution kernel length'
)
parser.add_argument(
    '-r', '--rate',
    action='store',
    dest='rate',
    default=100.0,
    type=float,
    help='Features sampling rate'
)
parser.add_argument(
    '-C', '--clip',
    action='store',
    dest='clip',
    default=0,
    type=float,
    help='Clip limit'
)

########################
# Evaluation arguments #
########################

parser.add_argument(
    '-gap', '--gap',
    action='store',
    dest='gap',
    default=0.02,
    type=float,
    help='Acceptable gap for evaluation'
)

parser.add_argument(
    '-R', '--results',
    action='store',
    dest='res_file',
    default='results.txt',
    type=str,
    help='Results file'
)
