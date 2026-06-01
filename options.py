import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=10, help="Number of local epochs")
    parser.add_argument('--local_steps', type=int, default=None, help="Alias for local_epoch; matches baseline naming")
    parser.add_argument('--global_epoch', type=int, default=50, help="Number of global epochs")
    parser.add_argument('--global_rounds', type=int, default=None, help="Alias for global_epoch; matches baseline naming")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--test_batch_size', type=int, default=256, help="Test batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader worker processes")

    parser.add_argument('--user_sample_rate', type=float, default=0.8, help="Sample rate for user sampling")
    parser.add_argument('--client_fraction', type=float, default=None, help="Alias for user_sample_rate; matches baseline naming")
    parser.add_argument('--seed', type=int, default=41, help="Random seed")

    parser.add_argument('--epsilon_file', type=str, default='gauss safety level', help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1e-5, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=1.0, help="Gradient clipping bound")
    parser.add_argument('--nm_decay', type=bool, default=True, help="noise_multiplier decay or not")
    parser.add_argument('--decay_factor', type=float, default=0.99, help="noise_multiplier decay factor")

    parser.add_argument('--fisher_threshold', type=float, default=0.4, help="Fisher information threshold for parameter selection")
    parser.add_argument('--fisher_max_batches', type=int, default=0, help="Maximum batches used to estimate Fisher information; 0 means all batches")
    parser.add_argument('--gamma', type=float, default=10.0, help="Layer-wise Fisher noise scaling hyper-parameter")
    parser.add_argument('--max_clip_norm', type=float, default=4.0, help="Maximum layer-wise clipped gradient norm cmax")
    parser.add_argument('--lambda_1', type=float, default=0.1, help="Lambda value for EWC regularization term")
    parser.add_argument('--lambda_2', type=float, default=0.05, help="Lambda value for regularization term to control the update magnitude")

    parser.add_argument('--device', type=int, default=0, help='Set the visible CUDA device for calculations')

    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum for local updates")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="SGD weight decay for local updates")

    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_noise', action='store_true')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'fmnist', 'cifar10', 'cifar100'])
    parser.add_argument('--data_dir', type=str, default=None, help="Root directory for datasets")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory for generated CSV results")
    parser.add_argument('--output_csv', type=str, default=None, help="Exact CSV output path; overrides output_dir")
    parser.add_argument('--partition', type=str, default=None, choices=['iid', 'non-iid', 'dirichlet'], help="Partition alias matching baseline")
    parser.add_argument('--iid', action='store_true', default=True, help="Use IID partition for supported datasets")
    parser.add_argument('--no-iid', dest='iid', action='store_false', help="Use non-IID partition for supported datasets")

    parser.add_argument('--dir_alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet_alpha', type=float, default=None, help="Alias for dir_alpha; matches baseline naming")

    parser.add_argument('--dirStr', type=str, default='')

    parser.add_argument('--store', type=bool, default=False)

    parser.add_argument('--appendix', type=str, default='')

    parser.add_argument('--fedavg', type=bool, default=False)
    parser.add_argument('--weiavg', type=bool, default=False)
    parser.add_argument('--deavg', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--phi', type=float, default=None, help="Alias for alpha; AdapL aggregation balance hyper-parameter")
    parser.add_argument('--eval_client_models', action='store_true', help="Evaluate each selected client model; baseline-style runs leave this off")



    args = parser.parse_args()
    if args.local_steps is not None:
        args.local_epoch = args.local_steps
    if args.global_rounds is not None:
        args.global_epoch = args.global_rounds
    if args.client_fraction is not None:
        args.user_sample_rate = args.client_fraction
    if args.dirichlet_alpha is not None:
        args.dir_alpha = args.dirichlet_alpha
    if args.phi is not None:
        args.alpha = args.phi
    if args.partition == 'iid':
        args.iid = True
    elif args.partition in {'non-iid', 'dirichlet'}:
        args.iid = False
    return args
