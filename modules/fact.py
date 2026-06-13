"""
FACT (F1 of Artifact/Content Terms) 指标

基于实体匹配的F1分数评估：提取预测文本和目标文本中的名词实体，计算精确率、召回率和F1。
参考: wsigen-master/utils/stat/metric/fact.py
"""
import re
import nltk

# 确保NLTK数据已下载
nltk.data.find('tokenizers/punkt')
nltk.data.find('taggers/averaged_perceptron_tagger')


def entity_match(pred, target):
    """
    计算预测文本和目标文本之间的实体匹配F1分数

    Args:
        pred: 预测文本 (str)
        target: 目标文本 (str)

    Returns:
        f1: F1分数，如果没有实体则返回 None
    """
    # 类型检查：确保输入是字符串
    if isinstance(pred, list):
        pred = ' '.join(str(x) for x in pred)
    if not isinstance(pred, str):
        pred = str(pred)
    if isinstance(target, list):
        target = ' '.join(str(x) for x in target)
    if not isinstance(target, str):
        target = str(target)

    if not pred.strip() or not target.strip():
        return None

    # 从预测文本提取实体（名词）
    entities = []
    sentences = nltk.sent_tokenize(pred)
    for sent in sentences:
        for word, tag in nltk.pos_tag(nltk.word_tokenize(sent)):
            if tag.startswith('NN'):
                clean_word = re.sub(r'([^a-z])', '', word.lower())
                if clean_word:
                    entities.append(clean_word)

    # 从目标文本提取实体
    entities_gt = []
    sentences = nltk.sent_tokenize(target)
    for sent in sentences:
        for word, tag in nltk.pos_tag(nltk.word_tokenize(sent)):
            if tag.startswith('NN'):
                clean_word = re.sub(r'([^a-z])', '', word.lower())
                if clean_word:
                    entities_gt.append(clean_word)

    # 去重
    entities = list(set(entities))
    entities_gt = list(set(entities_gt))

    # 如果没有实体，返回 None
    if len(entities) == 0 or len(entities_gt) == 0:
        return None

    # 计算精确率
    matches = sum(1 for e in entities if e in entities_gt)
    precision = matches / len(entities)

    # 计算召回率
    matches = sum(1 for e in entities_gt if e in entities)
    recall = matches / len(entities_gt)

    # 计算F1
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_fact_score(predictions, targets):
    """
    计算一批预测文本的FACT实体匹配分数

    Args:
        predictions: 预测文本列表 或 {idx: [text]} 字典
        targets: 目标文本列表 或 {idx: [text]} 字典

    Returns:
        fact_score: 平均F1分数 (float)
    """
    # 处理字典输入格式 {idx: [text]} -> list of texts
    if isinstance(predictions, dict):
        predictions = [predictions[i][0] for i in range(len(predictions))]
    if isinstance(targets, dict):
        targets = [targets[i][0] for i in range(len(targets))]

    scores = []
    for pred, tgt in zip(predictions, targets):
        score = entity_match(pred, tgt)
        if score is not None:
            scores.append(score)

    if len(scores) == 0:
        return 0.0

    return sum(scores) / len(scores)
