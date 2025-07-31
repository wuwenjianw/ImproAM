import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
import csv

from torch.utils.data import DataLoader
import time
from datetime import timedelta
import pickle
from utils.functions import parse_softmax_temperature
import matplotlib.pyplot as plt
mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)

def get_tasks(dataset, task_size):
    for i in range(dataset.size):
        data = dataset.data[i]
        loc = data['loc'][0:task_size]
        deadline = data['deadline'][0:task_size]
        dataset.data[i]['loc'] = loc
        dataset.data[i]['deadline'] = deadline

    return dataset


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model, opts)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:  # False
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        # dataset = get_tasks(dataset, task_size)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, task, durations, tours = zip(*results)  # Not really costs since they should be negative


    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))

    return results


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    # dataset.data['max_n_agents'] = 3

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size, shuffle=False)

    results = []
    tasks_done_total = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs, tasks_done = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                tasks_done_total.extend(tasks_done)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        i = 0
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME == "mrta":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append({"cost":cost, "tasks_done": tasks_done_total[i][0],"total_duration":duration, "sequence":seq})
            i +=1
    # plot tasks done here
    # plt.plot(tasks_done_total)
    # plt.show()
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--datasets", nargs='+', default=["ouput/data_100_3_0.2.pkl"], help="Filename of the dataset(s) to evaluate")
    # data/mrta/50_nodes_mrta.pkl
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=100,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--graph_size', type=int, default=100, help="The size of the problem graph")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                            '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', default="greedy", type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', default='', type=str)   
    # outputs/mrta_100_1000/run_20250522T110128  outputs/mrta_100_30000/run_20250618T130159
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')

    # LightCAPSGNN
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--k2', type=int, default=3)
    parser.add_argument("--features-dimensions", type=int, default=32,
                        help="node features dimensions. Default is 128.")
    parser.add_argument("--capsule-dimensions", type=int, default=8,
                        help="Capsule dimensions. Default is 4,6,8,10,12.")
    parser.add_argument("--capsule-num", type=int, default=10)
    parser.add_argument("--num-gcn-layers", type=int, default=3),
    parser.add_argument("--num-gcn-channels", type=int, default=2),
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of routing iterations. Default is 3.")
    parser.add_argument("--theta", type=float, default=0.1,
                        help="Reconstruction loss weight. Default is 0.1.")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help="Number of instances per batch during training")

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]

    # =======================新版本--dataset=======================
    # 指定目录路径
    # directory = 'ouput/test_data'

    # # 获取目录下所有pkl文件
    # pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    # output_dict = {'2': [], '3':[], '5': [], '7': []}  # , 'else': []
    # # 读取每个pkl文件
    # for file_name in pkl_files:
    #     file_path = os.path.join(directory, file_name)
    #     results = eval_dataset(file_path, 0, opts.softmax_temperature, parser.parse_args())
    #     if file_name[4] in output_dict.keys():
    #         output_dict[file_name[4]] += results
    #     else:
    #         output_dict['else'] += results
    
    # for output_list in output_dict.items():
    #     # 打开文件进行写入
    #     # CapAM,LightAM
    #     # dataset_result, random_result
    #     with open(f'ouput/dataset_result/LightAM/result_{output_list[0]}.txt', 'w') as file:
    #         for item in output_list[1]:
    #             file.write(str(item) + '\n')  # 每个元素一行写入

    #     import pickle
    #     with open(f'ouput/dataset_result/LightAM/result_{output_list[0]}.pkl', 'wb') as f:
    #         pickle.dump(output_list[1], f)


    # =======================新版本--random_dataset=======================
    # 指定目录路径
    # directory = 'ouput/test_new'

    # # 获取目录下所有pkl文件
    # pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    # output_dict = {'50_2': [], '50_3':[], '100_3': [], '100_5': [], '250_7':[], '250_10': [], '500_10': [], '500_20': []}  # , 'else': []
    # # 读取每个pkl文件
    # for file_name in pkl_files:
    #     print(file_name)
    #     file_path = os.path.join(directory, file_name)
    #     # results = eval_dataset(file_path, 0, opts.softmax_temperature, parser.parse_args())
    #     if file_name[5: 9] in output_dict.keys():
    #         opts.model = "outputs/mrta_50_30000/run_20250703T234538"
    #         opts.graph_size = 50
    #         opts.capsule_dimensions = 8
    #         results = eval_dataset(file_path, 0, opts.softmax_temperature, opts)
    #         output_dict[file_name[5: 9]] += results
    #     elif file_name[5: 10] in output_dict.keys():
    #         if file_name[5: 8] == '100':
    #             print("loading 100")
    #             opts.graph_size = 100
    #             opts.capsule_dimensions = 8
    #             opts.model = "outputs/mrta_100_30000/run_20250702T120330"
    #         elif file_name[5: 8] == '250':
    #             print("loading 250")
    #             opts.graph_size = 250
    #             opts.model = "outputs/mrta_250_30000/run_20250709T191403"
    #             opts.capsule_dimensions = 10
    #         results = eval_dataset(file_path, 0, opts.softmax_temperature, opts)
    #         output_dict[file_name[5: 10]] += results
    #     elif file_name[5: 11] in output_dict.keys():
    #         if file_name[5: 8] == '250':
    #             opts.graph_size = 250
    #             opts.model = "outputs/mrta_250_30000/run_20250709T191403"
    #             opts.capsule_dimensions = 10
    #         else:
    #             opts.graph_size = 500
    #             opts.capsule_dimensions = 8
    #             opts.model = "outputs/mrta_500_10000/run_20250703T235505"
    #         results = eval_dataset(file_path, 0, opts.softmax_temperature, opts)
    #         output_dict[file_name[5: 11]] += results
    
    # for output_list in output_dict.items():
    #     # 打开文件进行写入
    #     # CapAM,LightAM
    #     # dataset_result, random_result
    #     with open(f'ouput/random_result/ImproAM/result_{output_list[0]}.txt', 'w') as file:
    #         for item in output_list[1]:
    #             file.write(str(item) + '\n')  # 每个元素一行写入

    #     import pickle
    #     with open(f'ouput/random_result/ImproAM/result_{output_list[0]}.pkl', 'wb') as f:
    #         pickle.dump(output_list[1], f)
    #     print('endding')

    # =======================新版本--ablation=======================
    # 指定目录路径
    directory = 'ouput/test_ablation'

    # 获取目录下所有pkl文件
    model_list = {'p2c10': "outputs/mrta_100_30000/run_20250722T094047", 'p1c10': "outputs/mrta_100_30000/run_20250722T221422", 
                  'p3c6': "outputs/mrta_100_30000/run_20250701T213459", 'p3c8': "outputs/mrta_100_30000/run_20250702T120330", 
                  'p3c10': "outputs/mrta_100_30000/run_20250702T233602", 'p3c12': "outputs/mrta_100_30000/run_20250707T215132",}
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    for i,j in model_list.items():
        output_dir = "ouput/Ablation/" + i
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            print(f"文件夹 '{output_dir}' 创建成功")
        else:
            print(f"文件夹 '{output_dir}' 已存在")

        opts.model = j
        if len(i) == 4:
            # continue
            opts.capsule_dimensions = int(i[3])
            opts.n_encode_layers = int(i[1])
            opts.p = int(i[1])
        else:
            # if i[3:5] != "12":
            #     continue
            opts.capsule_dimensions = int(i[3:5])
            opts.n_encode_layers = int(i[1])      
            opts.p = int(i[1])
        
        output_dict = {'2': [], '3':[], '5': [], '7': []}  # , 'else': []
        # 读取每个pkl文件
        for file_name in pkl_files:
            file_path = os.path.join(directory, file_name)
            results = eval_dataset(file_path, 0, opts.softmax_temperature, opts)
            if file_name[9] in output_dict.keys():
                output_dict[file_name[9]] += results
            else:
                output_dict['else'] += results
        
        for output_list in output_dict.items():
            # 打开文件进行写入
            # CapAM,LightAM
            # dataset_result, random_result
            with open(f'{output_dir}/result_{output_list[0]}.txt', 'w') as file:
                for item in output_list[1]:
                    file.write(str(item) + '\n')  # 每个元素一行写入

            import pickle
            with open(f'{output_dir}/result_{output_list[0]}.pkl', 'wb') as f:
                pickle.dump(output_list[1], f)

    # =======================新版本--random_dataset (LightAM) =======================
    # 指定目录路径
    # directory = 'ouput/test_new'

    # # 获取目录下所有pkl文件
    # pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    # output_dict = {'50_2': [], '50_3':[], '100_3': [], '100_5': [], '250_7':[], '250_10': [], '500_10': [], '500_20': []}  # , 'else': []
    # # 读取每个pkl文件
    # for file_name in pkl_files:
    #     print(file_name)
    #     file_path = os.path.join(directory, file_name)
    #     # results = eval_dataset(file_path, 0, opts.softmax_temperature, parser.parse_args())
    #     if file_name[5: 9] in output_dict.keys():
    #         opts.model = "outputs/mrta_50_30000/run_20250711T154439"
    #         opts.graph_size = 50
    #         opts.capsule_dimensions = 10
    #         results = eval_dataset(file_path, 0, opts.softmax_temperature, opts)
    #         output_dict[file_name[5: 9]] += results
    #     elif file_name[5: 10] in output_dict.keys():
    #         if file_name[5: 8] == '100':
    #             print("loading 100")
    #             opts.graph_size = 100
    #             opts.capsule_dimensions = 8
    #             opts.model = "outputs/mrta_100_30000/run_20250618T130159"
    #         elif file_name[5: 8] == '250':
    #             print("loading 250")
    #             opts.graph_size = 250
    #             opts.model = "outputs/mrta_250_30000/run_20250710T223157"
    #             opts.capsule_dimensions = 10
    #         results = eval_dataset(file_path, 0, opts.softmax_temperature, opts)
    #         output_dict[file_name[5: 10]] += results
    #     elif file_name[5: 11] in output_dict.keys():
    #         if file_name[5: 8] == '250':
    #             opts.graph_size = 250
    #             opts.model = "outputs/mrta_250_30000/run_20250710T223157"
    #             opts.capsule_dimensions = 10
    #         else:
    #             opts.graph_size = 500
    #             opts.capsule_dimensions = 8
    #             opts.model = "outputs/mrta_500_10000/run_20250703T235505"
    #         results = eval_dataset(file_path, 0, opts.softmax_temperature, opts)
    #         output_dict[file_name[5: 11]] += results
    
    # for output_list in output_dict.items():
    #     # 打开文件进行写入
    #     # CapAM,LightAM
    #     # dataset_result, random_result
    #     with open(f'ouput/random_result/LightAM/result_{output_list[0]}.txt', 'w') as file:
    #         for item in output_list[1]:
    #             file.write(str(item) + '\n')  # 每个元素一行写入

    #     import pickle
    #     with open(f'ouput/random_result/LightAM/result_{output_list[0]}.pkl', 'wb') as f:
    #         pickle.dump(output_list[1], f)
    #     print('endding')

    # =======================老版本=======================  
    # all_files = "ouput/eval_50_3_0.2.pkl"# "data/mrta/100_nodes_mrta.pkl" ouput/data.pkl
    # file_n = open(all_files, 'rb')
    # datasets = pickle.load(file_n)
    # tot = []
    # for width in widths:
    #     for dataset_path in datasets:
    #         results = eval_dataset(all_files, width, opts.softmax_temperature, parser.parse_args())
    #         tot.append(results[0]['tasks_done'])

    #         # 打开文件进行写入
    #         with open('ouput/licapam.txt', 'w') as file:
    #             for item in results:
    #                 file.write(str(item) + '\n')  # 每个元素一行写入

    #         import pickle
    #         with open('ouput/licapam.pkl', 'wb') as f:
    #             pickle.dump(results, f)

    #         break
    #     break

    # with open('CapAM.csv', 'w') as f:
    #     write = csv.writer(f)
    #     write.writerows((np.array(tot).T).reshape((96,1)).tolist())