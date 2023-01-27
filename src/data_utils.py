import json
import logging
import os
import random

import torch
from torch.utils.data import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def shorten_words_and_labels(words, labels, limit: int):
    og_ent_token_num = len([i for i in labels if i != "O"])
    
    # TODO：最大前缀删除
    prefix_drop_num = len(words) - limit
    label_set = list(set(labels[:prefix_drop_num]))
    if "O" in label_set:  # 前面几个 token  没有 O
        label_set.remove("O")
        if not label_set:  # 只有 O，则放心舍去
            final_words, final_labels = words[prefix_drop_num:], labels[prefix_drop_num:]
            assert len([i for i in final_labels if i != "O"]) == og_ent_token_num, f"{words}\n{labels}"
            return final_words, final_labels
    
    # TODO：标点删除
    remain_len = len(words)
    final_words, final_labels = [], []
    _words, _labels = [], []
    # split long sentence:
    has_ent = False
    for i, (word, label) in enumerate(zip(words, labels)):
        if label != "O":  # 如果为内部实体，那么一定不能切分
            _words.append(word)
            _labels.append(label)
            has_ent = True
            continue
        
        # 如果是标点内部token，则加入缓存
        if word not in [",", ";", "?", "!", "(", ")", "."]:
            _words.append(word)
            _labels.append(label)
            continue
        
        # 如果是标点，那么就需要进行切分了
        else:
            # 如果缓存中有 entity，则需要保留
            if has_ent:
                final_words.extend(_words + [word])
                final_labels.extend(_labels + [label])
            
            if remain_len - len(_words) <= limit:
                final_words.extend(words[1 + i:])
                final_labels.extend(labels[1 + i:])
                break
            
            remain_len = remain_len - len(_words) - 1
            _words, _labels = [], []
            has_ent = False
    
    assert len([i for i in final_labels if i != "O"]) == og_ent_token_num, f"{words}\n{labels}"
    return final_words, final_labels


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, span_info):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.span_info = span_info


def read_and_load_data(args, mode, tokenizer):
    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    
    if os.path.exists(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth"):
        return torch.load(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth")
    
    data_path = os.path.join(args.source_dir, f"{mode}_{args.way_num}_{args.shot_num}.jsonl")
    logger.info("Reading tasks from {}...".format(data_path))
    with open(data_path, "r", encoding="utf-8") as json_file:
        json_list = list(json_file)
    
    output_tasks = []
    for task_id, json_str in enumerate(json_list):
        _support_features, _query_features = [], []
        support_features, query_features = [], []
        s_lens, q_lens = [], []
        
        task = json.loads(json_str)
        
        types = task["types"]
        types.append("O")
        type2id = {v: k for k, v in enumerate(types)}
        support = task["support"]
        
        for i, (words, labels) in enumerate(zip(support["word"], support["label"])):
            input_feature = convert_example_to_features(words, labels, type2id,
                                                        args.max_seq_length,
                                                        tokenizer,
                                                        cls_token_segment_id=0,
                                                        pad_token_segment_id=0, )
            
            support_features.append(input_feature)
            s_lens.append(len(input_feature[0]))
        
        max_len = max(s_lens)
        support_features = [pad_all(*f, max_len, pad_token_id) for f in support_features]
        
        query = task["query"]
        for i, (words, labels) in enumerate(zip(query["word"], query["label"])):
            input_feature = convert_example_to_features(words, labels, type2id,
                                                        args.max_seq_length,
                                                        tokenizer,
                                                        cls_token_segment_id=0,
                                                        pad_token_segment_id=0, )
            
            query_features.append(input_feature)
            q_lens.append(len(input_feature[0]))
        
        max_len = max(q_lens)
        query_features = [pad_all(*f, max_len, pad_token_id) for f in query_features]
        
        output_tasks.append({
            "type2id": type2id,
            "support_features": support_features,
            "query_features": query_features
        })
    
    if not os.path.exists('cache'):
        os.makedirs('cache')
    torch.save(output_tasks, f"cache/{mode}_{args.way_num}_{args.shot_num}.pth")
    
    return output_tasks


def convert_example_to_features(
        words, labels,
        label_map,
        max_seq_length,
        tokenizer,
        cls_token_segment_id=1,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    
    tokens = []
    label_ids = []
    spans = []
    
    last_label = None
    span_token_id = []
    if len(words) > 64:  # 过滤
        words, labels = shorten_words_and_labels(words, labels, 64)
    
    for i, (word, label) in enumerate(zip(words, labels)):
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:  # ! 有特殊字符报错("\u2063") 导致 word_tokens = []
            continue
        # if word_tokens[0] in ["and", "is", "are", "was", "'", "`", "(", ")", "the", "for", "to"]:
        #     continue
        
        tokens.extend(word_tokens)
        if label != "O":
            if span_token_id and (last_label != label):
                spans.append((span_token_id[0], span_token_id[-1], label_ids[-1]))
                span_token_id = []
            span_token_id.extend([1 + len(label_ids) + i for i in range(len(word_tokens))])  # + cls
        else:
            if span_token_id:
                spans.append((span_token_id[0], span_token_id[-1], label_ids[-1]))
                span_token_id = []
        
        label_ids.extend([label_map[label]] * len(word_tokens))
        last_label = label
    if span_token_id:
        spans.append((span_token_id[0], span_token_id[-1], label_ids[-1]))
    
    # Random sample O-span
    golden_idxs = [(s, e) for (s, e, i) in spans]
    O_num = 2 * int(0.2 * (len(tokens) + len(spans) * 5))
    
    for _ in range(O_num):
        while True:
            start_idx = random.randint(1, len(tokens))
            end_idx = random.randint(0, 5) + start_idx
            if end_idx > len(tokens):
                continue
            if (start_idx, end_idx) in golden_idxs:
                continue
            spans.append((start_idx, end_idx, label_map['O']))
            break
    
    spans = list(set(spans))
    
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        spans = [(s, e, i) for (s, e, i) in spans if e < (max_seq_length - special_tokens_count)]
    
    # 查看是否有被删除的 span
    if [(s, e, i) for (s, e, i) in spans if e >= (max_seq_length - special_tokens_count)]:
        print(words)
        exit()
    
    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens = [cls_token] + tokens
    label_ids = [pad_token_label_id] + label_ids
    segment_ids = [cls_token_segment_id] + segment_ids
    assert len(tokens) == len(label_ids), f"{words}"
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    
    assert len(input_ids) == len(input_mask) \
           == len(segment_ids) == len(label_ids)
    return input_ids, input_mask, segment_ids, label_ids, spans


def pad_all(input_ids, input_mask, segment_ids, label_ids, spans,
            max_seq_length,
            pad_token_id,
            mask_padding_with_zero=True,
            pad_token_segment_id=0,
            pad_token_label_id=-100,
            ):
    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += [pad_token_id] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length
    label_ids += [pad_token_label_id] * padding_length
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    
    return InputFeatures(input_ids=torch.tensor(input_ids, dtype=torch.long),
                         input_mask=torch.tensor(input_mask, dtype=torch.long),
                         segment_ids=torch.tensor(segment_ids, dtype=torch.long),
                         label_ids=torch.tensor(label_ids, dtype=torch.long),
                         span_info=spans)


class InstanceDataset(Dataset):
    def __init__(self, task, domain):
        self.features_list = task[f"{domain}_features"]
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        return self.features_list[idx]
    
    def collate(self, batch_tasks):
        batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids, batch_spans = [], [], [], [], []
        for _feature in batch_tasks:
            batch_input_ids.append(_feature.input_ids)
            batch_input_mask.append(_feature.input_mask)
            batch_segment_ids.append(_feature.segment_ids)
            batch_label_ids.append(_feature.label_ids)
            batch_spans.append(_feature.span_info)
        
        batch_input_ids = torch.stack(batch_input_ids)
        batch_input_mask = torch.stack(batch_input_mask)
        batch_segment_ids = torch.stack(batch_segment_ids)
        batch_label_ids = torch.stack(batch_label_ids)
        
        return batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids, batch_spans
