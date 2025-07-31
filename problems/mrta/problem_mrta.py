from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.mrta.state_mrta import StateMRTA
from utils.beam_search import beam_search


class MRTA(object):
    NAME = 'mrta'  # Capacitated Vehicle Routing Problem

    # VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    # 计算路径成本
    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size, loc_vec_size = dataset['loc'].size()
        # print(batch_size, graph_size, loc_vec_size)
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
                       torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                       sorted_pi[:, -graph_size:]
               ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))
        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        cost = (
                       (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                       + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
                       + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
               ), None

        return cost

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MRTADataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMRTA.initialize(*args, **kwargs)

    # 启发式搜索
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)  # 计算上下文全局信息

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = MRTA.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class MRTADataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0,
                n_depot=1,
                initial_size=None,
                deadline_min=None,
                deadline_max=None,
                n_agents=20,
                max_speed=.1,
                distribution=None):
        super(MRTADataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            # ================= new ===========================
            # 遍历列表中的每个字典
            for item in data:
                # 检查 'depot' 参数是否存在
                if 'depot' not in item:
                    # 添加 'depot' 参数，您可以根据需要设置其值
                    item['depot'] = torch.tensor([[0.0, 0.0]])  # 默认值，可以自定义
                    item['max_n_agents'] =  item['n_agents']# torch.tensor([[10]])
            # ================= end ===========================

            self.data = data # [make_instance(args) for args in data[offset:offset+num_samples]]

        else:
            # ================= new ===========================
            # max_n_agent = n_agents # 10
            # n_agents_available = n_agents # torch.tensor([2, 3, 5, 7])
            # # 随机选择每个样本的机器人数量
            # agents_ids = torch.randint(0, 4, (num_samples, 1))
            # # 随机生成的组数
            # groups = torch.randint(1, 3, (num_samples, 1))
            # #  随机生成的数
            # dist = torch.randint(1, 5, (num_samples, 1))
            # data = []
            # for i in range(num_samples):
                # n_agents = n_agents # n_agents_available[agents_ids[i, 0].item()].item()
                # # 随机小车的位置
                # agents_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100)
                # # 随机生成任务的位置
                # loc = torch.FloatTensor(size, 2).uniform_(0, 1)
                # # 工作负载
                # workload = torch.FloatTensor(size).uniform_(.2, .2)
                
                # # 计算任务的最小和最大截止时间
                # # 先计算任务位置和机器人位置之间的最大欧几里得距离，再根据速度推算出最短的执行时间
                # d_low = (((loc[:, None, :].expand((size, max_n_agent, 2)) - agents_location[None].expand(
                #     (size, max_n_agent, 2))).norm(2, -1).max() / max_speed) + 20).to(torch.int64) + 1
                # d_high = ((35) * (45) * 100 / (380) + d_low).to(torch.int64) + 1
                # d_low = d_low * (.5 * groups[i, 0])
                # d_high = ((d_high * (.5 * groups[i, 0]) / 10).to(torch.int64) + 1) * 10
                # deadline_normal = (torch.rand(size, 1) * (d_high - d_low) + d_low).to(torch.int64) + 1
                # # 正常任务
                # n_norm_tasks = dist[i, 0] * 25
                # # 随机生成一个rand_mat，然后根据分布k_th_quant从中选择正常任务，并通过normal_dist_tasks标记这些任务
                # rand_mat = torch.rand(size, 1)
                # k = n_norm_tasks.item()  # For the general case change 0.25 to the percentage you need
                # k_th_quant = torch.topk(rand_mat.T, k, largest=False)[0][:, -1:]
                # bool_tensor = rand_mat <= k_th_quant
                # normal_dist_tasks = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))
                # # 根据normal_dist_tasks生成松弛任务slack_tasks
                # slack_tasks = (normal_dist_tasks - 1).to(torch.bool).to(torch.int64)
                # normal_dist_tasks_deadline = normal_dist_tasks * deadline_normal
                # slack_tasks_deadline = slack_tasks * d_high
                # # 计算任务的最终截止时间
                # deadline_final = normal_dist_tasks_deadline + slack_tasks_deadline
                # # 生成小车的起始位置和工作能力
                # robots_start_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100).to(
                #     device=deadline_final.device)
                # # 随机生成机器人工作能力，工作能力值在[0.01, 0.01] 之间
                # robots_work_capacity = torch.randint(2, 3, (max_n_agent, 1), dtype=torch.float,
                #                                     device=deadline_final.device).view(-1) / 100
                # case_info = {
                #     'loc': loc,
                #     'depot': torch.FloatTensor(1, 2).uniform_(0, 1),
                #     'deadline': deadline_final.to(torch.float).view(-1),
                #     'workload': workload,
                #     'initial_size': 100,
                #     'n_agents': torch.tensor([[n_agents]]),
                #     'max_n_agents': torch.tensor([[max_n_agent]]),
                #     'max_speed': max_speed,
                #     'robots_start_location': robots_start_location,
                #     'robots_work_capacity': robots_work_capacity
                # }
                # data.append(case_info)
            # ================= end ===========================

            # ================= old ===========================
            # task100: 2,3,5,7
            # task250: 3, 5, 7, 10
            # task500: 5, 7, 10, 20
            max_n_agent = 10

            if size == 250:
                n_agents_available = torch.tensor([3,5,7,10])
            elif size == 500:
                n_agents_available = torch.tensor([5,7,10,20])
            else:
                n_agents_available = torch.tensor([2,3,5,7])

            agents_ids = torch.randint(0, 4, (num_samples, 1))
            groups = torch.randint(1, 3, (num_samples, 1))
            dist = torch.randint(1, 5, (num_samples, 1))
            data = []
            for i in range(num_samples):
                n_agents = n_agents_available[agents_ids[i, 0].item()].item()
                agents_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100)
                loc = torch.FloatTensor(size, 2).uniform_(0, 1)
                workload = torch.FloatTensor(size).uniform_(.2, .2)
                d_low = (((loc[:, None, :].expand((size, max_n_agent, 2)) - agents_location[None].expand(
                    (size, max_n_agent, 2))).norm(2, -1).max() / max_speed) + 20).to(torch.int64) + 1
                d_high = ((35) * (45) * 100 / (380) + d_low).to(torch.int64) + 1
                d_low = d_low * (.5 * groups[i, 0])
                d_high = ((d_high * (.5 * groups[i, 0]) / 10).to(torch.int64) + 1) * 10
                deadline_normal = (torch.rand(size, 1) * (d_high - d_low) + d_low).to(torch.int64) + 1

                # 正常任务
                n_norm_tasks = dist[i, 0] * int( size / 4)

                rand_mat = torch.rand(size, 1)
                k = n_norm_tasks.item()  # For the general case change 0.25 to the percentage you need
                k_th_quant = torch.topk(rand_mat.T, k, largest=False)[0][:, -1:]
                bool_tensor = rand_mat <= k_th_quant
                normal_dist_tasks = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0))
                slack_tasks = (normal_dist_tasks - 1).to(torch.bool).to(torch.int64)
                normal_dist_tasks_deadline = normal_dist_tasks * deadline_normal
                slack_tasks_deadline = slack_tasks * d_high
                deadline_final = normal_dist_tasks_deadline + slack_tasks_deadline
                robots_start_location = (torch.randint(0, 101, (max_n_agent, 2)).to(torch.float) / 100).to(
                    device=deadline_final.device)
                robots_work_capacity = torch.randint(1, 3, (max_n_agent, 1), dtype=torch.float,
                                                    device=deadline_final.device).view(-1) / 100
                case_info = {
                    'loc': loc,
                    'depot': torch.FloatTensor(1, 2).uniform_(0, 1),
                    'deadline': deadline_final.to(torch.float).view(-1),
                    'workload': workload,
                    'initial_size': 100,
                    'n_agents': torch.tensor([[n_agents]]),
                    'max_n_agents': torch.tensor([[max_n_agent]]),
                    'max_speed': max_speed,
                    'robots_start_location': robots_start_location,
                    'robots_work_capacity': robots_work_capacity
                }
                data.append(case_info)

            self.data = data

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]