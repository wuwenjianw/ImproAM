import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
import time

from nets.graph_encoder import  GCAPCN
from nets.capsgnn import CapsGNN

from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        # return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                embedding_dim,
                hidden_dim,
                problem,
                n_encode_layers=2,
                tanh_clipping=10.,
                mask_inner=True,
                mask_logits=True,
                normalization='batch',
                n_heads=8,
                checkpoint_encoder=False,
                shrink_size=None, opts=False):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None   # 解码方式
        self.temp = 1.0
        self.is_mrta = problem.NAME == 'mrta'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size   # 用于剪枝

        torch.autograd.set_detect_anomaly(True)

        self.robots_state_query_embed = nn.Linear(3, embedding_dim)
        self.robot_taking_decision_query = nn.Linear(3, embedding_dim)

        node_dim = 3

        if self.is_mrta:
            step_context_dim_new = embedding_dim + embedding_dim + 1

        # Special embedding projection for depot node
        # 仓库节点的特殊嵌入投影
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.init_embed = nn.Linear(node_dim, embedding_dim)  # 普通节点的嵌入

        # 嵌入部分修改
        self.falg = 1

        if self.falg == 0:
            self.embedder = GCAPCN()
        elif self.falg == 1:
            self.embedder = CapsGNN(opts, 4)

        # 注意力机制相关层
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        # 对于每个节点，我们计算（glimpse key，glimpse value，logit key），因此 3 * embedding_dim

        # 用于计算注意力机制的 glimpse key, glimpse value, logit key
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # 用于计算固定上下文信息（整个图的全局信息）
        # if self.falg == 0:
        #     # 用于计算注意力机制的 glimpse key, glimpse value, logit key
        #     self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        #     self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # elif self.falg == 1:
        #     self.project_node_embeddings = nn.Linear(36, 3 * embedding_dim, bias=False)
        #     self.project_fixed_context = nn.Linear(36, embedding_dim, bias=False)

        # 计算 解码过程中的动态上下文
        self.project_step_context = nn.Linear(step_context_dim_new, embedding_dim, bias=False)
        
        #  用于当前节点的上下文投影（可能用于路径决策）
        self.project_context_cur_loc = nn.Linear(2, embedding_dim, bias=False)

        assert embedding_dim % n_heads == 0

        # 将注意力机制的输出投影回原始维度
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    # 训练时选择解码方式
    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            import time
            # start_time = time.time()
            # embeddings, _ = self.embedder(self._init_embed(input))
            embeddings, _ = self.embedder(input)  # 【2，101，128】
            # end_time = time.time() - start_time

        # 计算路径
        _log_p, pi, cost = self._inner(input, embeddings)
        # cos, mask = self.problem.get_costs(input, pi)
        mask = None

        # 计算路径的对数似然（log likelihood），用于强化学习中的策略梯度更新
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    # 启发式搜索
    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    # 计算固定特征
    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    # 计算最优扩展
    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)  # 获取选择动作的对数概率

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)  # 计算对数似然

    def _init_embed(self, input):

        features = ('deadline',)

        return torch.cat(
            (
                self.init_embed_depot(input['depot'])[:, None, :],
                self.init_embed(torch.cat((
                    input['loc'],
                    *(input[feat][:, :, None] for feat in features)
                ), -1))
            ),
            1
        )
        # TSP

    def _inner_eval(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0

        time_sl = []
        while not (self.shrink_size is None and not (state.all_finished().item() == 0)):
            start_time = time.time()

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            log_p, mask = self._get_log_p(fixed, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            end_time1 = time.time() - start_time
            state = state.update(selected)

            # print(state.robots_list)

            start_time2 = time.time()

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1
            end_time2 = time.time() - start_time2
            time_sl.append(end_time1 + end_time2)


        cost = (torch.div(state.tasks_finish_time, state.deadline) * (
                    torch.div(state.tasks_finish_time, state.deadline) > 1).to(torch.int64)).sum(-1)

        return torch.stack(outputs, 1), torch.stack(sequences, 1), cost, state.tasks_done_success.tolist()

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        # 执行解码步骤
        i = 0


        # initial tasks
        while not (self.shrink_size is None and not (state.all_finished().item() == 0)):

            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            # Only the required ones goes here, so we should
            #  We need a variable that track which all tasks are available
            # log_p:[b, 1, 101] mask:[b, 1, 101]  只有必需的任务才会出现在这里，因此我们应该使用一个变量来跟踪所有可用的任务
            log_p, mask = self._get_log_p(fixed, state)  # 

            # 选择序列中下一个节点的索引, long log_p.exp()转换为实际概率
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = state.update(selected)

            # 现在制作log_p，通过“取消收缩”选择所需的输出大小 Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        cost = (torch.div(state.tasks_finish_time, state.deadline) * (
                    torch.div(state.tasks_finish_time, state.deadline) > 1).to(torch.int64)).sum(-1)

        return torch.stack(outputs, 1), torch.stack(sequences, 1), cost

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner_eval(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(input)[0]),  # Pack input with embeddings (additional input) - for CCN encoding
            # (input, self.embedder(self._init_embed(input))[0]), ## for MHA encoding
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)  # 即选择概率最高的节点
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        # 为了提高效率，图嵌入的固定上下文投影仅计算一次 [3, 128]
        # 对embeddings输出进行一次线性层，整和为[3, 101, 128]

        graph_embed = embeddings.mean(1)  # [b, 128]

        #
        # 固定上下文 = (batch_size, 1, embed_dim)，以便使用并行时间步进行广播[3, 1, 128]
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # 注意力节点嵌入的投影是预先计算一次的，logit为专门给选择网络算logits的“key”
        # [b,1, 101, 128]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # 将key和value进行重新整形，使其适用于多头注意力
        # 由于只有一个头，因此无需重新排列logit的键
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),  #(n_heads, b, num_steps, graph_size, 16：head_dim)
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous() # .contiguous()占据连续的内存空间
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]  # 最后生成的是一个三维张量
        )

    def _get_log_p(self, fixed, state, normalize=True):

        # Compute query = context node embedding
        # 计算查询 = 上下文节点嵌入 [3,1,128]
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings,
                                                                        state))

        # Compute keys and values for the nodes
        # [8, b, 1, 101, 16] [8, b, 1, 101, 16] [b, 1, 101, 128] 计算键和值
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        # [b, 1, 101] 计算掩码
        mask = state.get_mask()

        # [b, 1, 101]    Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        if normalize:  # 归一化
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        返回每一步的上下文，可选择一次返回多个步骤（为了有效评估模型）
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()  # [3,1]
        batch_size, num_steps = current_node.size()
        # Embedding of previous node + remaining capacity

        robots_current_destination = state.robots_current_destination.clone()  # [3,10]

        working_robots = ((state.robots_initial_decision_sequence <= (state.n_agents - 1)).to(torch.float)).to(
            device=robots_current_destination.device)  # [3,10]

        # 确保仅保留正在工作的机器人的信息  [3,10,3]
        current_robot_states = torch.cat(
            (state.robots_current_destination_location, state.robots_work_capacity[:, :, None]), -1) * working_robots[:,
                                                                                                    :, None]
        
        # 获取当前正在做决策的机器人的目标位置和工作容量  [3, 3]
        decision_robot_state = torch.cat((state.robots_current_destination_location[
                                            state.ids, state.robot_taking_decision].view(batch_size, -1),
                                        state.robots_work_capacity[state.ids, state.robot_taking_decision]),
                                        -1) 

        # 嵌入  [3,128]
        robots_states_embedding = self.robots_state_query_embed(current_robot_states).sum(-2)
        # 决策机器人嵌入计算 [3,128]
        decision_robot_state_embedding = self.robot_taking_decision_query(decision_robot_state)

        return torch.cat((state.next_decision_time[:, :, None], decision_robot_state_embedding[:, None],
                        robots_states_embedding[:, None]), -1)  # [3,1,257]

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # 变形query以适配多头注意力 [8, b, 1, 1, 16] [head_num, b, num_steps, 1, val_size]
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # 计算注意力 [8, b, 1, 1, 101]
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf  # 把不可选位置的compatibility赋值为-inf，这样在softmax计算时，概率会变成0（完全屏蔽）

        # [8, b, 1, 1, 16] (n_heads, batch_size, num_steps, val_size) softmax归一化，得到注意力分布,使用注意力权重对glimpse_V 进行加权求和，得到最终的Value表征
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # [b, 1, 1, 128] (batch_size, num_steps, 1, embedding_dim) 计算glimpse(注意力上下文) 
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # [b, 1, 1, 128] 现在不需要投射glimpse，因为它可以被吸收到project_out中
        final_Q = glimpse

        # [b, 1, 101] 计算logits（未归一化概率） logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping  # 限制logits的范围，防止梯度爆炸
        if self.mask_logits:
            logits[mask] = -math.inf  # 屏蔽不可选位置

        return logits, glimpse.squeeze(-2)  # 表示最终的注意力上下文信息

    def _get_attention_node_data(self, fixed, state):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        # 将输入v重新整形，使其适用于多头注意力计算
        # n_heads作为第一个维度，使得计算时可以并行处理多个头的注意力
        # batch_size 作为第二个维度，保持批处理结构
        # num_steps 作为第三个维度，控制时间步长
        # graph_size 作为第四个维度，表示图的节点数
        # head_dim 作为最后一个维度，表示单个注意力头的特征维度
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )