import argparse
import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


def read_config(path):
    return Config.load(path)

def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip norm')

    parser.add_argument('--max_len', type=int, default=40,
                        help='gradient clip norm')
    
    parser.add_argument('--optim', type=str, default='adam',
                        choices=['adam', 'amsgrad', 'adagrad'],
                        help='optimizer')
    parser.add_argument('--loss_fn', type=str, default='triplet',
                        choices=['triplet', 'cosine', 'custom_triplet'],
                        help='loss function')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--freeze', type=bool, default=False)

    parser.add_argument('--seed', type=int, default=2804,
                        help='Random seed')
    parser.add_argument('--PRE_TRAINED_MODEL_NAME', default='allenai/biomed_roberta_base',
                        help='huggingface model name')

    # Data placeholder
    parser.add_argument('--train_dir', default='./data/triplet_data.csv')
    parser.add_argument('--val_dir', default='./data/test_stroke.csv')
    parser.add_argument('--save_dir', default='./ckpt/')
    parser.add_argument('--log_dir', default='./log/log.txt')
    
    parser.add_argument('--train_threshold', default=0.5)
    parser.add_argument('--val_threshold', default=0.5)
    parser.add_argument('--print_every', default=30)
    
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)