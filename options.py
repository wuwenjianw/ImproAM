import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")
    # Data
    parser.add_argument('--problem', default='mrta', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=100, help="The size of the problem graph")
    parser.add_argument('--initial_size', type=int, default=150, help="The size of the problem graph when the simulation starts")
    parser.add_argument('--batch_size', type=int, default=128, help="Number of instances per batch during training")
    parser.add_argument('--epoch_size', type=int, default=30000, help="Number of instances per epoch during training")
    parser.add_argument('--n_agents', type=int, default=10, help="Number of robots")
    parser.add_argument('--n_depot', type=int, default=1, help="Number of depot")
    parser.add_argument('--agent_max_speed', type=int, default=.01, help="Max speed for the robot")
    parser.add_argument('--deadline_min', type=int, default=40,
                        help="Min value for deadline")
    parser.add_argument('--deadline_max', type=int, default=550,
                        help="Max value for deadline")

    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                            'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--lr_model', type=float, default=0.0002, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    # 指数移动平均基线衰减
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    # 预热基线的周期数,只能与rollout baseline一起使用
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                            'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                            ' to save memory (default None means no shrinking)')
    # 训练期间使用的数据分布、默认值和选项取决于问题
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    #从第0个时期开始（与学习率衰减相关）
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    # LightCAPSGNN
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--k2', type=int, default=3)
    parser.add_argument("--features-dimensions", type=int, default=32,
                        help="node features dimensions. Default is 128.")
    
    parser.add_argument("--p", type=int, default=1)    
    parser.add_argument("--capsule-dimensions", type=int, default=10,
                        help="Capsule dimensions. Default is 4,6,8,10,12.")
    
    parser.add_argument("--capsule-num", type=int, default=10)
    parser.add_argument("--num-gcn-layers", type=int, default=3),
    parser.add_argument("--num-gcn-channels", type=int, default=2),
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of routing iterations. Default is 3.")
    parser.add_argument("--theta", type=float, default=0.1,
                        help="Reconstruction loss weight. Default is 0.1.")
    parser.add_argument('--dropout', type=float, default=0)

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}_{}".format(opts.problem, opts.graph_size, opts.epoch_size),
        opts.run_name
    )

    return opts