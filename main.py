import argparse
from model.configs import Config, str2bool
from model.data_loader import get_loader
from model.solver import Solver


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', type=str, default='exp1', help='Creates the dir to save results')
    parser.add_argument('--epochs', type = int, default = 50, help = 'the number of training epochs')
    parser.add_argument('--lr', type = float, default = 5e-5, help = 'the learning rate')
    parser.add_argument('--l2_reg', type = float, default = 1e-4, help = 'l2 regularizer')
    parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'the dropout ratio')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'the batch size')
    parser.add_argument('--tag', type = str, default = 'dev', help = 'A tag for experiments')
    parser.add_argument('--ckpt_path', type = str, default = 'None', help = 'checkpoint path for inference or weight initialization')
    parser.add_argument('--clip', type=float, default=5.0, help='Max norm of the gradients')
    parser.add_argument('--train', type=str2bool, default='True', help='when use Train')
    parser.add_argument('--seed', type=int, default=12345, help='Chosen seed for generating random numbers')
    parser.add_argument('--init_type', type=str, default="xavier", help='Weight initialization method')
    parser.add_argument('--init_gain', type=float, default=None, help='Scaling factor for the initialization methods')

    parser.add_argument('--input_size', type=int, default=512, help='Feature size expected in the input')
    parser.add_argument('--text_size', type=int, default=512, help='Feature size expected in the input')
    parser.add_argument('--pos_enc', type=str2bool, default=True, help="If positional encoding will be used")
    parser.add_argument('--heads', type=int, default=8, help="Number of heads for the cross-attention module")
    parser.add_argument('--annotations', type=int, default=10, help="Number of annotations per video (in S-VideoXum = 10, in S-NewsVSum = 1)")

    parser.add_argument('--dataset', type=str, default='S_VideoXum', help="The name of the dataset: ['S_VideoXum' | 'S_NewsVSum']")
    parser.add_argument('--split_num', type=int, default=0, help="To be used for S_NewsVSum: [0:4]")


    opt = parser.parse_args()

    kwargs = vars(opt)
    config = Config(**kwargs)

    print(f"Running experiment: {config.exp_num}")

    train_loader = get_loader('train', dataset=config.dataset, split_num=config.split_num)
    val_loader = get_loader('val', dataset=config.dataset, split_num=config.split_num)
    if config.dataset == 'S_VideoXum':
        test_loader = get_loader('test', dataset=config.dataset)
    elif config.dataset == 'S_NewsVSum':
        test_loader = get_loader('val', dataset=config.dataset, split_num=config.split_num)
    else:
        raise ValueError("Error: no valid dataset. Must be: ['S_VideoXum' | 'S_NewsVSum']")

    solver = Solver(config, train_loader, val_loader, test_loader)

    solver.build()
    test_model_ckpt_path = None

    if config.train:
        best_f1_ckpt_path = solver.train()
        solver.test(best_f1_ckpt_path)
    else:
        test_model_ckpt_path = config.ckpt_path
        if test_model_ckpt_path is None:
            print("Trained model checkpoint requried. Exit program")
            exit()
        else:
            solver.test(test_model_ckpt_path)