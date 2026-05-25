import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=10, help="Number of local epochs")
    parser.add_argument('--global_epoch', type=int, default=50, help="Number of global epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")

    parser.add_argument('--user_sample_rate', type=float, default=0.8, help="Sample rate for user sampling")

    parser.add_argument('--epsilon_file', type=str, default='gauss safety level', help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1e-5, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=1.0, help="Gradient clipping bound")
    parser.add_argument('--nm_decay', type=bool, default=True, help="noise_multiplier decay or not")
    parser.add_argument('--decay_factor', type=float, default=0.99, help="noise_multiplier decay factor")

    parser.add_argument('--fisher_threshold', type=float, default=0.4, help="Fisher information threshold for parameter selection")
    parser.add_argument('--lambda_1', type=float, default=0.1, help="Lambda value for EWC regularization term")
    parser.add_argument('--lambda_2', type=float, default=0.05, help="Lambda value for regularization term to control the update magnitude")

    parser.add_argument('--device', type=int, default=0, help='Set the visible CUDA device for calculations')

    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")

    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_noise', action='store_true')

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--iid', action='store_true', default=True, help="Use IID partition for supported datasets")
    parser.add_argument('--no-iid', dest='iid', action='store_false', help="Use non-IID partition for supported datasets")

    parser.add_argument('--dir_alpha', type=float, default=0.3)

    parser.add_argument('--dirStr', type=str, default='')

    parser.add_argument('--store', type=bool, default=False)

    parser.add_argument('--appendix', type=str, default='')

    parser.add_argument('--fedavg', type=bool, default=False)
    parser.add_argument('--weiavg', type=bool, default=False)
    parser.add_argument('--deavg', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.8)



    args = parser.parse_args()
    return args
