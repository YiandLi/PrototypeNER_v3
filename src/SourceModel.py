import logging

import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import LpDistance
from torch import nn

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceModel(nn.Module):
    def __init__(self, args, encoder):
        super(SourceModel, self).__init__()
        
        self.encoder = encoder
        self.args = args
        
        self.head = nn.Linear(encoder.config.hidden_size, args.down_size)
        self.tail = nn.Linear(encoder.config.hidden_size, args.down_size)
        
        # self.downsize_layer = nn.Sequential(
        #     nn.Linear(2 * encoder.config.hidden_size, args.down_size),  # entity_representation
        # )
        
        self.distance_metric = LpDistance(power=2)
        
        self.entity_classifier = nn.Sequential(
            nn.Linear(args.down_size * 2, 1)
        )
    
    def _euclidean_metric(self, a, b, normalize=False):
        if normalize:
            a = F.normalize(a)
            b = F.normalize(b)
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        distance = ((a - b) ** 2).sum(dim=2)
        return distance
    
    def forward_(self, all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_span_infos,
                 O_id, post_avg_by_class=False, prior_avg_by_class=False, only_retur_protos=False):
        
        device = all_input_ids.device
        # TODO：得到所有实体向量和对应 id
        return_sentence_tuples = self.get_entities_and_ids(all_input_ids, all_input_mask, all_segment_ids,
                                                           all_span_infos, O_id)
        
        if only_retur_protos:
            total_entity_vectors, total_entity_ids = None, None
            for (entity_vectors, entity_ids) in return_sentence_tuples:
                if total_entity_vectors == None:
                    total_entity_vectors, total_entity_ids = entity_vectors, entity_ids
                else:
                    total_entity_vectors = torch.vstack((total_entity_vectors, entity_vectors))
                    total_entity_ids = torch.hstack((total_entity_ids, entity_ids))
            
            assert len(total_entity_vectors) == len(total_entity_ids)
            return torch.tensor(0., requires_grad=True), total_entity_vectors, total_entity_ids
        
        # if not (entity_ids != O_id).int().sum() > 0:  # batch 内部没有实体
        #     # logger.info("There's no valid entity in this batch.")
        #     return torch.tensor(0., requires_grad=True), entity_vectors, entity_ids
        
        # TODO：计算损失
        # 计算 loss
        not_ent_loss = torch.tensor(0., requires_grad=True).to(device)
        is_ent_loss = torch.tensor(0., requires_grad=True).to(device)
        for i, (entity_vectors, entity_ids) in enumerate(return_sentence_tuples):
            entity_pros = self.entity_classifier(entity_vectors)
            is_entity_pros = entity_pros[entity_ids != O_id].reshape(-1)
            not_entity_pros = entity_pros[entity_ids == O_id].reshape(-1)
            
            is_entity_pros = torch.hstack((is_entity_pros, torch.tensor(0., requires_grad=False).to(device)))
            not_entity_pros = torch.hstack((not_entity_pros, torch.tensor(0., requires_grad=False).to(device)))
            
            # print(f"{i} :pred_ent_num/pred_not_ent_num - {(entity_pros > 0).sum() / (entity_pros < 0).sum():.3f}")
            
            # Circle Loss
            not_ent_loss = not_ent_loss + torch.logsumexp(not_entity_pros, dim=-1)
            is_ent_loss = is_ent_loss + torch.logsumexp(-is_entity_pros, dim=-1)
        
        logger.info(f"\t\t\tnot_ent_loss: {not_ent_loss:.3f}, is_ent_loss:{is_ent_loss:.3f}")
        loss = not_ent_loss / 5 + is_ent_loss
        return loss, entity_vectors, entity_ids
    
    def gather_entity_vectors_by_class(self, entity_vectors, entity_ids):
        ent_protos = {int(i): [] for i in set(entity_ids.tolist())}
        for i, ent_vec in zip(entity_ids, entity_vectors):
            ent_protos[int(i)].append(ent_vec)
        for i in ent_protos.keys():
            ent_protos[int(i)] = torch.stack(ent_protos[i]).mean(0)
        return torch.stack(list(ent_protos.values())), torch.tensor(list(ent_protos.keys()))
    
    def get_proto_vectors(self, O_id,
                          entity_vectors, entity_ids,
                          prior_avg_by_class, post_avg_by_class
                          ):
        # TODO：按照实体类别进行 avg pooling
        avg_proto_entity_vectors, avg_proto_entity_ids = self.gather_entity_vectors_by_class(entity_vectors,
                                                                                             entity_ids)
        # TODO：得到 O-span 和 non-O-span 向量和对应 id ( non-O-span 是否作二次平均处理
        if prior_avg_by_class:  # prior intro and inter-class average
            no_ent_vector = avg_proto_entity_vectors[avg_proto_entity_ids == O_id].squeeze()
            ent_avg_vector = avg_proto_entity_vectors[avg_proto_entity_ids != O_id].mean(0)  # shot=1时，不要用
        else:
            no_ent_vector = avg_proto_entity_vectors[avg_proto_entity_ids == O_id].squeeze()
            ent_avg_vector = entity_vectors[entity_ids != O_id].mean(0)
        
        # TODO：得到原型向量和对应 id（ 是否进行平均
        if post_avg_by_class:
            proto_entity_vectors, proto_entity_ids = \
                avg_proto_entity_vectors[avg_proto_entity_ids != O_id], \
                avg_proto_entity_ids[avg_proto_entity_ids != O_id]
        else:
            proto_entity_vectors, proto_entity_ids = \
                entity_vectors[entity_ids != O_id], entity_ids[entity_ids != O_id]
        
        return no_ent_vector, ent_avg_vector, proto_entity_vectors, proto_entity_ids
    
    def get_loss(self, ent_ids, ent_dis, proto_entity_ids, is_ent_pro=None):
        """
         ent_ids: size ：golden entity id
         ent_dis: size ：distance to the prototypes [golden_entity_num, support_entity_num]
         proto_entity_ids： prototype id [] length of  support_entity_num
         is_ent_pro:  prior probability  [golden_entity_num, 1]
        """
        # 求分母
        ent_dis_pro = torch.softmax(-ent_dis, -1)  # golden_entity_num,support_entity_num
        # 得到目标后验
        ent_num = ent_ids.shape[0]
        proto_num = proto_entity_ids.shape[0]
        ent_ids = ent_ids[:, None].expand(ent_num, proto_num)  # golden_entity_num,support_entity_num
        all_pro = ent_dis_pro[ent_ids == proto_entity_ids]
        # 先验
        if is_ent_pro == None:
            return torch.sum(-torch.log(all_pro))  # 没有先验
        else:
            is_ent_pro = is_ent_pro.expand(ent_num, proto_num)
            is_ent_pro = is_ent_pro[ent_ids == proto_entity_ids]
            return torch.sum(-torch.log(is_ent_pro * all_pro))  # 联合
    
    def get_entities_and_ids(self, all_input_ids, all_input_mask, all_segment_ids,
                             all_span_infos, O_id, get_all_negs=True):
        """
        得到所有合法的实体 vector 和对应的 golden_entity_id
        
        get_all_negs: support集合 可以选择设置为 False，但是其他集合必须是 True
        """
        
        entity_vector = []
        entity_ids = []
        
        batch_outputs = self.encoder(all_input_ids, all_input_mask, all_segment_ids)
        last_hidden_state = batch_outputs.last_hidden_state
        
        assert last_hidden_state.shape[0] == len(all_span_infos)
        
        heads = self.head(last_hidden_state)
        tails = self.tail(last_hidden_state)
        
        if not get_all_negs:
            for i, span_info in enumerate(all_span_infos):
                for s, e, entity_id in span_info:
                    span_representation = torch.cat((heads[i][s], tails[i][e]))
                    entity_vector.append(span_representation)
                    entity_ids.append(torch.tensor(entity_id, dtype=float))
            entity_vectors = torch.stack(entity_vector)
            entity_ids = torch.stack(entity_ids)
        
        else:
            batch_size, seq_len, hidden_state = heads.shape
            raw_extend = heads.unsqueeze(2).expand(-1, -1, seq_len, -1)
            col_extend = tails.unsqueeze(1).expand(-1, seq_len, -1, -1)
            entity_vector = torch.cat([raw_extend, col_extend], 3)
            entity_ids = torch.zeros(batch_size, seq_len, seq_len) + O_id
            for i, span_info in enumerate(all_span_infos):
                for s, e, entity_id in span_info:
                    if entity_id != O_id:
                        entity_ids[i, s, e] = entity_id
            
            # 构造上三角 mask
            _mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=0)
            return_sentence_tuples = []
            # padding mask: cls and sep
            valid_token_len = [len(torch.where(i != 0)[0]) for i in all_input_ids]
            for i, (single_mask, _token_len) in enumerate(zip(_mask, valid_token_len)):
                single_mask[0] = 0
                single_mask[:, 0] = 0
                single_mask[_token_len - 1:] = 0  # [sep]
                single_mask[:, _token_len - 1:] = 0
                
                _mask = single_mask.reshape(- 1)
                _entity_ids = entity_ids[i].reshape(-1)
                _entity_vector = entity_vector[i].reshape(-1, hidden_state * 2)
                
                _entity_ids = _entity_ids[_mask == 1]
                _entity_vectors = _entity_vector[_mask == 1]
                return_sentence_tuples.append((_entity_vectors, _entity_ids))
            # _mask = _mask.reshape(batch_size, - 1)
            # entity_ids = entity_ids.reshape(batch_size, -1)
            # entity_vector = entity_vector.reshape(batch_size, -1, hidden_state * 2)
            
            # entity_ids = entity_ids[_mask == 1]
            # entity_vectors = entity_vector[_mask == 1]
        
        # assert len(entity_ids) == len(entity_vectors)
        # return entity_vectors, entity_ids
        
        return return_sentence_tuples
    
    def get_pred_vector(self, all_input_ids, all_input_mask, all_segment_ids):
        batch_outputs = self.encoder(all_input_ids, all_input_mask, all_segment_ids, )
        last_hidden_state = batch_outputs.last_hidden_state
        heads = self.head(last_hidden_state)
        tails = self.tail(last_hidden_state)
        
        batch_size, seq_len, hidden_state = heads.shape
        raw_extend = heads.unsqueeze(2).expand(-1, -1, seq_len, -1)
        col_extend = tails.unsqueeze(1).expand(-1, seq_len, -1, -1)
        entity_reps = torch.cat([raw_extend, col_extend], 3)
        
        return entity_reps
