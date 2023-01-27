import copy
import logging

import torch
from pytorch_metric_learning.distances import LpDistance
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data_utils import InstanceDataset
from src.utils import get_evaluate_fpr

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

distance_metric = LpDistance(power=2)


def test_meta(args, device, model, target_task):
    label2id = target_task['type2id']
    support_dataset = InstanceDataset(target_task, domain="support")
    query_dataset = InstanceDataset(target_task, domain="query")
    support_dataloader = DataLoader(support_dataset, batch_size=args.inner_batch_size,
                                    collate_fn=support_dataset.collate,
                                    shuffle=False, drop_last=False,
                                    num_workers=16 if device == 'cuda' else 0)
    query_dataloader = DataLoader(query_dataset, batch_size=args.inner_batch_size,
                                  collate_fn=query_dataset.collate,
                                  shuffle=False, drop_last=False,
                                  num_workers=16 if device == 'cuda' else 0)
    
    # Test
    O_id = label2id["O"]
    optimizer, scheduler = set_optimizer(args, model, train_steps=len(support_dataloader), get_scheduler=False)
    
    # TODO: inner train on support set
    best_support_loss = 1e99
    for support_epoch in range(args.inner_identify_loops):
        for batch in support_dataloader:
            support_loss = 0.
            batch_support_input_ids, batch_support_input_mask, \
            batch_support_segment_ids, batch_support_label_ids = (i.to(device) for i in batch[:4])
            batch_support_spans = batch[-1]
            
            step_loss, _, _ = model.forward_(batch_support_input_ids, batch_support_input_mask,
                                             batch_support_segment_ids, batch_support_label_ids,
                                             batch_support_spans, O_id,
                                             post_avg_by_class=args.post_avg_by_class,
                                             prior_avg_by_class=args.prior_avg_by_class)
            
            support_loss += step_loss.item()
            optimizer.zero_grad()
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            if scheduler:
                scheduler.step()
        if support_loss <= best_support_loss:
            best_support_loss = support_loss
        else:
            logger.info(f"This Dev process use {support_epoch + 1} support epochs, final loss {best_support_loss:.3f}")
            break
    
    support_entity_vectors, support_entity_ids = None, None
    # TODO: get prototype vectors
    for batch in support_dataloader:
        batch_support_input_ids, batch_support_input_mask, \
        batch_support_segment_ids, batch_support_label_ids = (i.to(device) for i in batch[:4])
        batch_support_spans = batch[-1]
        with torch.no_grad():
            _, entity_vectors, entity_ids = model.forward_(batch_support_input_ids, batch_support_input_mask,
                                                           batch_support_segment_ids, batch_support_label_ids,
                                                           batch_support_spans, O_id,
                                                           post_avg_by_class=args.post_avg_by_class,
                                                           prior_avg_by_class=args.prior_avg_by_class,
                                                           only_retur_protos=True)
        
        # 汇总
        if support_entity_vectors == None:
            support_entity_vectors = entity_vectors.to('cpu')
            support_entity_ids = entity_ids.to('cpu')
        else:
            support_entity_vectors = torch.vstack((support_entity_vectors, entity_vectors.to('cpu')))
            support_entity_ids = torch.hstack((support_entity_ids, entity_ids.to('cpu')))
    
    _, _, proto_entity_vectors, proto_entity_ids = \
        model.get_proto_vectors(O_id, support_entity_vectors, support_entity_ids,
                                args.prior_avg_by_class, args.post_avg_by_class)
    
    # TODO: get tag prediction on query set
    for batch in query_dataloader:
        batch_query_input_ids, batch_query_input_mask, \
        batch_query_segment_ids, batch_query_label_ids = (i.to(device) for i in batch[:4])
        batch_query_spans = batch[-1]
        with torch.no_grad():
            _, entity_vectors, golden_entity_ids = model.forward_(batch_query_input_ids, batch_query_input_mask,
                                                                  batch_query_segment_ids, batch_query_label_ids,
                                                                  batch_query_spans, O_id,
                                                                  post_avg_by_class=args.post_avg_by_class,
                                                                  prior_avg_by_class=args.prior_avg_by_class,
                                                                  only_retur_protos=True)
        
        hidden_size = entity_vectors.shape[-1]
        
        # prior: 先判断是不是 entity
        is_entity_prob = model.entity_classifier(entity_vectors).reshape(-1).to('cpu')
        
        # 然后判断实体类型
        pred_vectors = distance_metric(entity_vectors.view(-1, hidden_size).to('cpu'), proto_entity_vectors)
        pred_ids = proto_entity_ids[torch.argmin(pred_vectors, dim=-1)]  # b_s, seq_len, seq_len, 1
        # 非实体设置为 O-tag
        pred_ids[is_entity_prob > 0] = O_id
    
    prior_f1, prior_precision, prior_recall, ent_f1, ent_precision, ent_recall \
        = get_evaluate_fpr(pred_ids, golden_entity_ids, O_id=O_id)
    return prior_f1, prior_precision, prior_recall, ent_f1, ent_precision, ent_recall


def train_meta(args, label2id, train_task, model, device):
    """
    task_dataloader is ccomposed of three part:
        support tasks
        query tasks
        label map shared

    This function is composed of:
        inner train on support set ( negative sample for O-tag)
        get prototype vectors of support set ( negative sample for O-tag)
        get loss on query set ( full sample for candidate span)
    """
    
    support_dataset = InstanceDataset(train_task, domain="support")
    query_dataset = InstanceDataset(train_task, domain="query")
    support_dataloader = DataLoader(support_dataset, batch_size=args.inner_batch_size,
                                    collate_fn=support_dataset.collate,
                                    shuffle=False, drop_last=False,
                                    num_workers=16 if device == 'cuda' else 0)
    query_dataloader = DataLoader(query_dataset, batch_size=args.inner_batch_size,
                                  collate_fn=query_dataset.collate,
                                  shuffle=False, drop_last=False,
                                  num_workers=16 if device == 'cuda' else 0)
    
    O_id = label2id["O"]
    
    # TODO: inner train for entity identify
    optimizer, scheduler = set_optimizer(args, model, train_steps=len(support_dataloader), get_scheduler=False)
    
    for i in range(args.inner_identify_loops):  # 每个 task 训练多少轮
        
        # TODO: 每一轮都更新模型
        for support_batch, query_batch in zip(support_dataloader, query_dataloader):
            # TODO：存在问题，不应该用 batch level 进行 forward_，
            #  这样 不保证forward_内部包含所有vector
            
            batch_support_input_ids, batch_support_input_mask, \
            batch_support_segment_ids, batch_support_label_ids = (i.to(device) for i in support_batch[:4])
            batch_support_spans = support_batch[-1]
            
            batch_query_input_ids, batch_query_input_mask, \
            batch_query_segment_ids, batch_query_label_ids = (i.to(device) for i in query_batch[:4])
            batch_query_spans = query_batch[-1]
            
            s_step_loss, _, _ = model.forward_(batch_support_input_ids, batch_support_input_mask,
                                               batch_support_segment_ids, batch_support_label_ids,
                                               batch_support_spans, O_id,
                                               post_avg_by_class=args.post_avg_by_class,
                                               prior_avg_by_class=args.prior_avg_by_class)
            
            q_step_loss, _, _ = model.forward_(batch_query_input_ids, batch_query_input_mask,
                                               batch_query_segment_ids, batch_query_label_ids,
                                               batch_query_spans, O_id,
                                               post_avg_by_class=args.post_avg_by_class,
                                               prior_avg_by_class=args.prior_avg_by_class)
            
            one_step_loss = s_step_loss + q_step_loss
            
            optimizer.zero_grad()
            one_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            if scheduler:
                scheduler.step()
    
    # TODO: get model params for identify update
    _, identify_updated_params = get_names_and_params(model)
    identify_updated_params = copy.deepcopy(identify_updated_params)
    logger.info(f"\t\tidentify loss: {one_step_loss.item():.3f}")
    return identify_updated_params, None
    
    support_entity_vectors, support_entity_ids = None, None
    # TODO: get prototype vectors
    for batch in support_dataloader:
        batch_support_input_ids, batch_support_input_mask, \
        batch_support_segment_ids, batch_support_label_ids = (i.to(device) for i in batch[:4])
        batch_support_spans = batch[-1]
        entity_vectors, entity_ids = model.get_entities_and_ids(batch_support_input_ids,
                                                                batch_support_input_mask,
                                                                batch_support_segment_ids,
                                                                batch_support_spans, O_id)
        
        # 汇总
        if support_entity_vectors == None:
            support_entity_vectors = entity_vectors
            support_entity_ids = entity_ids
        else:
            support_entity_vectors = torch.vstack((support_entity_vectors, entity_vectors))
            support_entity_ids = torch.hstack((support_entity_ids, entity_ids))
    
    _, _, proto_entity_vectors, proto_entity_ids = \
        model.get_proto_vectors(O_id, support_entity_vectors, support_entity_ids,
                                args.prior_avg_by_class, args.post_avg_by_class)
    
    proto_entity_vectors = proto_entity_vectors.detach()  # 尝试 2，见下文
    # TODO: get loss for entity classification
    optimizer, scheduler = set_optimizer(args, model, train_steps=len(support_dataloader), get_scheduler=False)
    
    for i in range(args.inner_classify_loops):
        for batch in query_dataloader:
            batch_query_input_ids, batch_query_input_mask, \
            batch_query_segment_ids, batch_query_label_ids = (i.to(device) for i in batch[:4])
            batch_query_spans = batch[-1]
            
            # 序列方式
            q_entity_vectors, q_entity_ids = model.get_entities_and_ids(batch_query_input_ids,
                                                                        batch_query_input_mask,
                                                                        batch_query_segment_ids,
                                                                        batch_query_spans, O_id)
            
            # get loss of Entity classification
            if args.use_prior:
                entity_pros = model.entity_classifier(q_entity_vectors)
                is_entity_pros = entity_pros[q_entity_ids != O_id]
            else:
                is_entity_pros = None
            
            is_ent_vectors = q_entity_vectors[q_entity_ids != O_id]
            is_ent_ids = q_entity_ids[q_entity_ids != O_id]
            is_ent_dis = distance_metric(is_ent_vectors, proto_entity_vectors)
            is_ent_loss = model.get_loss(is_ent_ids, is_ent_dis, proto_entity_ids, is_entity_pros)
            
            # update
            optimizer.zero_grad()
            is_ent_loss.backward(retain_graph=True)
            """
            出现 BUG，计算图被破坏，因为 backward() 调用之后，图就被释放了。 proto_entity_vectors 无法回传了。
            
                尝试 1 # retain_graph=True 因为第一次 loop 后 proto_entity_vectors 的计算图被破坏了
                
                又出现错误 one of the variables needed for gradient computation has been modified by an inplace operation
                是因为首次更新后 模型改变了，所以 `proto_entity_vectors` 不能回传了
                
                尝试 2 ，时间快的做法： # 直接对 proto_entity_vectors 进行 detach
                尝试 3 ，每次重新得到 proto_entity_vectors
            """
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
    
    # TODO: get model params for classifier update
    _, classify_updated_params = get_names_and_params(model)
    classify_updated_params = copy.deepcopy(classify_updated_params)
    return identify_updated_params, classify_updated_params


def load_gradients(model, names, grads, norm=1):
    model_params = model.state_dict(keep_vars=True)
    for n, g in zip(names, grads):
        if model_params[n].grad is None:
            continue
        model_params[n].grad.data.add_(g.data / norm)  # accumulate normed grad


def update_weight(model, names, all_tasks_grad, step_len=1.):
    model_params = model.state_dict(keep_vars=True)
    norm = float(len(all_tasks_grad))
    for task_grad in all_tasks_grad:  # each task
        for n, g in zip(names, task_grad):
            if model_params[n].grad is None:
                continue
        model_params[n].data.add_(step_len * g / norm)


def load_weights(model, names, params):
    model_params = model.state_dict()
    for n, p in zip(names, params):
        model_params[n].data.copy_(p.data)


def get_names_and_params(model):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    return names, params


def set_optimizer(args, model, train_steps=None, get_scheduler=False):
    if not args.froze_encoder:
        # encoder 的 named_parameters()
        encoder_param_optimizer = list(model.encoder.named_parameters())
    
    # classifier/dense 的 named_parameters()
    dense_para_optimizer = list(model.head.named_parameters()) \
                           + list(model.tail.named_parameters()) \
                           + list(model.entity_classifier.named_parameters())
    
    # 筛选 pooler
    if not args.froze_encoder:
        encoder_param_optimizer = [n for n in encoder_param_optimizer if 'pooler' not in n[0]]
    dense_para_optimizer = [n for n in dense_para_optimizer if 'pooler' not in n[0]]
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    if not args.froze_encoder:
        optimizer_grouped_parameters += [
            {'params': [p for n, p in encoder_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01,
             'lr': args.encoder_lr},
            {'params': [p for n, p in encoder_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': args.encoder_lr}
        ]
    optimizer_grouped_parameters += [
        {'params': [p for n, p in dense_para_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01,
         'lr': args.downsize_lr
         },
        {'params': [p for n, p in dense_para_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.downsize_lr}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.encoder_lr, eps=1e-8)
    
    scheduler = None
    if get_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.01 * train_steps), num_training_steps=train_steps
        )
        logger.info(f"num_warmup_steps: {int(0.1 * train_steps)}")
    return optimizer, scheduler


def froze_model(args, model):
    # Froze model's PLM
    if args.froze_encoder:
        model.encoder.requires_grad = False
        for p in model.encoder.parameters():
            p.requires_grad = False
    
    # Embedding and Pooling freezing
    no_grad_param_names = ["embeddings", "pooler"]
    logger.info("The frozen parameters are:")
    for name, param in model.named_parameters():
        if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
            param.requires_grad = False
    
    return model
