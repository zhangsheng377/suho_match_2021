#!/usr/bin/env python
# coding: utf-8

# # 导包

# In[1]:


import sys 
sys.version


# In[2]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[3]:


# !unzip sohu2021_open_data_clean.zip
# !unzip chinese_L-12_H-768_A-12.zip


# In[4]:


# !pip install transformers


# In[5]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import sys
import re
from collections import Counter
import random
import json
from joblib import dump, load
from functools import partial
from datetime import datetime
import multiprocessing

from tqdm import tqdm
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
# tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
# import keras
from tensorflow.keras.metrics import top_k_categorical_accuracy, binary_accuracy
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model, load_model, model_from_json
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, binary_crossentropy
# from keras.utils import multi_gpu_model
# from keras.utils.training_utils import multi_gpu_model
# from tensorflow.keras.utils import multi_gpu_model
from transformers import (
    BertTokenizer,
    TFBertForPreTraining,
    TFBertModel,
)
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
import torch
from pyhanlp import *
import jieba

from my_utils import calculate_bm25_similarity, calculate_tf_cosine_similarity, calculate_tfidf_cosine_similarity


# In[6]:


tf.__version__


# In[7]:


keras.__version__


# In[8]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[9]:


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# In[ ]:





# In[10]:


data_path = "sohu2021_open_data_clean/"
# train_file_names = ["train.txt", "valid.txt", "round2.txt", "round3.txt"]
train_file_name = "data/shuffle_total_file.json"
text_max_length = 512
bert_path = r"chinese_L-12_H-768_A-12"

check_point_path = 'trained_model_substract_1/multi_keras_bert_sohu.weights'
weights_path = "trained_model_substract_1/multi_keras_bert_sohu_final.weights"
config_path = "trained_model_substract_1/multi_keras_bert_sohu_final.model_config.json"
result_path = "trained_model_substract_1/multi_keras_bert_sohu_test_result_final.csv"


# In[11]:


# bm25Model = load("bm25.bin")
# bm25Model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


# 转换bert模型，到pytorch的pd格式


# In[13]:


# !transformers-cli convert --model_type bert \
#   --tf_checkpoint chinese_L-12_H-768_A-12/bert_model.ckpt \
#   --config chinese_L-12_H-768_A-12/bert_config.json \
#   --pytorch_dump_output chinese_L-12_H-768_A-12/pytorch_model.bin


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 多任务分支模型

# ## 构建数据迭代器

# In[14]:


label_type_to_id = {'labelA':0, 'labelB':1}
label_to_id = {'0':0, '1':1}


# In[15]:


# def get_text_iterator(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             yield line


# In[16]:


def _transform_text(text):
   text = text.strip().replace('\n', '。').replace('\t', '').replace('\u3000', '')
   return re.sub(r'。+', '。', text)


# In[17]:


# def get_summary(text, senc_num=20):
#     a = HanLP.extractSummary(text, 20)
#     a_ = str(a)
#     return a_[1:-1]


# In[18]:


# def get_data_iterator(data_path, file_names):
#     # TODO: 随机取
#     file_iters = []
#     for file_name in file_names:
#       for category in os.listdir(data_path):
#           category_path = os.path.join(data_path, category)
#           if not os.path.isdir(category_path):
#               continue
              
#           file_path = os.path.join(category_path, file_name)
#           if not os.path.isfile(file_path):
#               continue
              
          
#           file_iter = get_text_iterator(file_path)
#           cat_source = 0
#           if category[0] == '长':
#             cat_source = 1
#           cat_target = 0
#           if category[1] == '长':
#             cat_target = 1
#           file_iters.append((file_iter, cat_source, cat_target))
        
#     while len(file_iters) > 0:
#         i = random.randrange(len(file_iters))
#         line = next(file_iters[i][0], None)
#         cat_source = file_iters[i][1]
#         cat_target = file_iters[i][2]
#         if line is None:
#             del file_iters[i]
#             continue
            
#         data = json.loads(line)

#         data['source'] = _transform_text(data['source'])
#         if len(data['source']) == 0:
#             print('source:', line, data)
#             break
# #                     continue

#         data['target'] = _transform_text(data['target'])
#         if len(data['target']) == 0:
#             print('target:', line, data)
#             break
# #                     continue

#         label_name_list = list(key for key in data.keys() if key[:5]=='label')
#         if len(label_name_list) != 1:
#             print('label_name_list:', line, data)
#             break
# #                     continue
#         label_name = label_name_list[0]
#         if data[label_name] not in label_to_id.keys():
#             print('label_name:', line, data, label_name)
#             break
# #                     continue
        
#         label_dict = {key: -1 for key in label_type_to_id.keys()}
#         label_dict[label_name] = label_to_id[data[label_name]]
#         if label_dict['labelA'] == 0:
#             label_dict['labelB'] = 0
#         if label_dict['labelB'] == 1:
#             label_dict['labelA'] = 1

#         yield data['source'], data['target'], cat_source, cat_target, label_dict['labelA'], label_dict['labelB']


# In[19]:


# it = get_data_iterator(data_path, train_file_names)


# In[20]:


# next(it)


# In[21]:


def get_sample_num(data_path, file_names):
    count = 0
    it = get_data_iterator(data_path, file_names)
    for data in tqdm(it):
        count += 1
    return count


# In[22]:


# sample_count = get_sample_num(data_path, train_file_names)
# sample_count


# In[23]:


# def get_shuffle_total_file(data_path, file_names, output_file_path):
#     data_list = []
#     it = get_data_iterator(data_path, file_names)
#     for source, target, cat_source, cat_target, labelA, labelB in tqdm(it):
#         json_data = {
#             'source':source,
#             'target':target,
#             'cat_source':cat_source,
#             'cat_target':cat_target,
#             'labelA':labelA,
#             'labelB':labelB
#         }
#         data_list.append(json_data)
#     random.shuffle(data_list)
    
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         for json_data in data_list:
#             output_file.write(f"{json.dumps(json_data)}\n")


# In[24]:


# get_shuffle_total_file(data_path, train_file_names, train_file_name)


# In[25]:


def get_data_iterator(data_path, file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line)
            yield json_data['source'], json_data['target'], json_data['cat_source'], json_data['cat_target'], json_data['labelA'], json_data['labelB']


# In[26]:


it = get_data_iterator(data_path, train_file_name)


# In[27]:


next(it)


# In[28]:


# sample_count = get_sample_num(data_path, train_file_name)
# sample_count


# In[29]:


def get_sample_y(data_path, file_names):
    labelA_list = []
    labelB_list = []
    it = get_data_iterator(data_path, file_names)
    for source, target, cat_source, cat_target, labelA, labelB in tqdm(it):
        if labelA != -1:
          labelA_list.append(labelA)
        if labelB != -1:
          labelB_list.append(labelB)
    return labelA_list, labelB_list


# In[30]:


# np.unique(labelA_list), labelA_list


# In[31]:


labelA_list, labelB_list = get_sample_y(data_path, train_file_name)
labelA_class_weights = class_weight.compute_class_weight('balanced', np.unique(labelA_list), np.array(labelA_list))
labelB_class_weights = class_weight.compute_class_weight('balanced', np.unique(labelB_list), np.array(labelB_list))
labelA_class_weights, labelB_class_weights


# In[32]:


tokenizer = BertTokenizer.from_pretrained(bert_path)


# In[33]:


def _get_indices(text, text_pair=None):
    return tokenizer.encode_plus(text=text,
                            text_pair=text_pair,
                            max_length=text_max_length, 
                            add_special_tokens=True, 
                            padding='max_length', 
#                             truncation_strategy='longest_first', 
                            truncation=True,
#                                          return_tensors='tf',
                            return_token_type_ids=True
                            )


# In[34]:


def get_keras_bert_iterator_notwhile(data_path, file_names, tokenizer):
    data_it = get_data_iterator(data_path, file_names)
    for source, target, cat_source, cat_target, labelA, labelB in data_it:
        data_source = _get_indices(text=source)
        data_target = _get_indices(text=target)
#             print(indices, type(indices), len(indices))
        seg_source = jieba.lcut(source)
        seg_target = jieba.lcut(target)
        bm25 = calculate_bm25_similarity(bm25Model, seg_source, seg_target)
        tf_cosine = calculate_tf_cosine_similarity(seg_source, seg_target)
        tfidf_cosine = calculate_tfidf_cosine_similarity(seg_source, seg_target, bm25Model.idf)
        id = ""
        yield data_source['input_ids'], data_source['token_type_ids'], data_source['attention_mask'],               data_target['input_ids'], data_target['token_type_ids'], data_target['attention_mask'],               bm25, tf_cosine, tfidf_cosine,               cat_source, cat_target,               labelA, labelB, id


# In[35]:


it = get_keras_bert_iterator_notwhile(data_path, train_file_name, tokenizer)


# In[36]:


# next(it)


# In[37]:


def to_tfrecord(it, output_path):
#     it = get_keras_bert_iterator_notwhile(data_path, train_file_name, tokenizer)
    with tf.io.TFRecordWriter(output_path) as tfrecord_writer:
        for ids_texta, token_type_ids_texta, attention_mask_texta,             ids_textb, token_type_ids_textb, attention_mask_textb,             bm25, tf_cosine, tfidf_cosine,             cat_texta, cat_textb,             labelA, labelB, id in tqdm(it):

            """ 2. 定义features """
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'ids_texta': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=ids_texta)),
                        'token_type_ids_texta': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=token_type_ids_texta)),
                        'attention_mask_texta': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=attention_mask_texta)),
                        'ids_textb': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=ids_textb)),
                        'token_type_ids_textb': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=token_type_ids_textb)),
                        'attention_mask_textb': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=attention_mask_textb)),
                        'bm25': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[bm25])),
                        'tf_cosine': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[tf_cosine])),
                        'tfidf_cosine': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[tfidf_cosine])),
                        'cat_texta': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[cat_texta])),
                        'cat_textb': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[cat_textb])),
                        'labelA': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[labelA])),
                        'labelB': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[labelB])),
                        'id': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[id.encode('utf-8')])),
                    }))

            """ 3. 序列化,写入"""
            serialized = example.SerializeToString()
            tfrecord_writer.write(serialized)


# In[38]:


# it = get_keras_bert_iterator_notwhile(data_path, train_file_name, tokenizer)
# to_tfrecord(it, "data/shuffle_total_file.tfrecord")


# In[39]:


# it = get_test_keras_bert_iterator(data_path, "test_with_id.txt")
# to_tfrecord(it, "data/test_file.tfrecord")


# In[40]:


def parse_from_single_example(example_proto, need_id):
    """ 从example message反序列化得到当初写入的内容 """
    # 描述features
    desc = {
        'ids_texta': tf.io.FixedLenFeature([512], dtype=tf.int64),
        'token_type_ids_texta': tf.io.FixedLenFeature([512], dtype=tf.int64),
        'attention_mask_texta': tf.io.FixedLenFeature([512], dtype=tf.int64),
        'ids_textb': tf.io.FixedLenFeature([512], dtype=tf.int64),
        'token_type_ids_textb': tf.io.FixedLenFeature([512], dtype=tf.int64),
        'attention_mask_textb': tf.io.FixedLenFeature([512], dtype=tf.int64),
        'bm25': tf.io.FixedLenFeature([1], dtype=tf.float32),
        'tf_cosine': tf.io.FixedLenFeature([1], dtype=tf.float32),
        'tfidf_cosine': tf.io.FixedLenFeature([1], dtype=tf.float32),
        'cat_texta': tf.io.FixedLenFeature([1], dtype=tf.int64),
        'cat_textb': tf.io.FixedLenFeature([1], dtype=tf.int64),
        'labelA': tf.io.FixedLenFeature([1], dtype=tf.int64),
        'labelB': tf.io.FixedLenFeature([1], dtype=tf.int64),
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
    }
    # 使用tf.io.parse_single_example反序列化
    example = tf.io.parse_single_example(example_proto, desc)
    
    data = {
        'ids_texta': example['ids_texta'],
        'token_type_ids_texta': example['token_type_ids_texta'],
        'attention_mask_texta': example['attention_mask_texta'],
        'ids_textb': example['ids_textb'],
        'token_type_ids_textb': example['token_type_ids_textb'],
        'attention_mask_textb': example['attention_mask_textb'],
        'bm25': example['bm25'],
        'tf_cosine': example['tf_cosine'],
        'tfidf_cosine': example['tfidf_cosine'],
        'cat_texta': example['cat_texta'],
        'cat_textb': example['cat_textb'],
    }
    label = {
        'labelA': example['labelA'],
        'labelB': example['labelB'],
    }
    if not need_id:
        return data, label
    return data, label, example['id']


# In[41]:


dataset = tf.data.TFRecordDataset(["data/test_file.tfrecord"])


# In[42]:


data_iter = iter(dataset)
first_example = next(data_iter)


# In[43]:


data = parse_from_single_example(first_example, need_id=False)


# In[44]:


# data['ids_texta'].numpy(), data['ids_texta'].numpy().shape


# In[45]:


def get_dataset(file_list, batch_size, epochs=None, need_id=False, options=None):
    dataset = tf.data.TFRecordDataset(file_list)
    if options:
        dataset = dataset.with_options(options)
    dataset = dataset.map(partial(parse_from_single_example, need_id=need_id), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if epochs:
        dataset = dataset.repeat(epochs)
    else:
        dataset = dataset.repeat()
    return dataset.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)                   .batch(batch_size)                   .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[46]:


dataset = get_dataset("data/shuffle_total_file.tfrecord", epochs=1, batch_size=1)
dataset


# In[47]:


# next(iter(dataset))


# In[48]:


def get_dataset_length(file_list):
    dataset = get_dataset(file_list, epochs=1, batch_size=1)
    count = 0
    for _ in tqdm(iter(dataset)):
        count += 1
    return count


# In[49]:


train_dataset_len = get_dataset_length("data/shuffle_total_file.tfrecord")
train_dataset_len


# In[ ]:





# In[ ]:





# In[50]:


def save_test_result(weights_path, result_file):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    it = get_dataset("data/test_file.tfrecord", epochs=1, batch_size=1, need_id=True, options=options)
    
    model = get_model()
    model.load_weights(weights_path)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"id,label\n")
        for data, _, id in tqdm(it):
#             print(id.numpy(), id.numpy()[0], str(id.numpy()[0], encoding="utf-8"), id.numpy()[0][-1], str(id.numpy()[0], encoding="utf-8")[-1])
            id_ = str(id.numpy()[0], encoding="utf-8")
            last_char = id_[-1]
            predict = model.predict(data)
            if last_char == 'a':
                predict_cls = 1 if predict['labelA'][0][0] > 0.5 else 0
            elif last_char == 'b':
                predict_cls = 1 if predict['labelB'][0][0] > 0.5 else 0
            else:
                print(id)
                continue
            f.write(f"{id_},{predict_cls}\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 定义模型

# In[51]:


def variant_focal_loss(gamma=2., alpha=0.5, rescale = False):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        # print(y_true)
        """
        Focal loss for bianry-classification
        FL(p_t)=-rescaled_factor*alpha_t*(1-p_t)^{gamma}log(p_t)
        
        Notice: 
        y_pred is probability after sigmoid

        Arguments:
            y_true {tensor} -- groud truth label, shape of [batch_size, 1]
            y_pred {tensor} -- predicted label, shape of [batch_size, 1]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})  
            alpha {float} -- (default: {0.5})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9  
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        model_out = tf.clip_by_value(y_pred, epsilon, 1.-epsilon)  # to advoid numeric underflow
        
        # compute cross entropy ce = ce_0 + ce_1 = - (1-y)*log(1-y_hat) - y*log(y_hat)
        ce_0 = tf.multiply(tf.subtract(1., y_true), -tf.math.log(tf.subtract(1., model_out)))
        ce_1 = tf.multiply(y_true, -tf.math.log(model_out))

        # compute focal loss fl = fl_0 + fl_1
        # obviously fl < ce because of the down-weighting, we can fix it by rescaling
        # fl_0 = -(1-y_true)*(1-alpha)*((y_hat)^gamma)*log(1-y_hat) = (1-alpha)*((y_hat)^gamma)*ce_0
        fl_0 = tf.multiply(tf.pow(model_out, gamma), ce_0)
        fl_0 = tf.multiply(1.-alpha, fl_0)
        # fl_1= -y_true*alpha*((1-y_hat)^gamma)*log(y_hat) = alpha*((1-y_hat)^gamma*ce_1
        fl_1 = tf.multiply(tf.pow(tf.subtract(1., model_out), gamma), ce_1)
        fl_1 = tf.multiply(alpha, fl_1)
        fl = tf.add(fl_0, fl_1)
        f1_avg = tf.reduce_mean(fl)
        
        if rescale:
            # rescale f1 to keep the quantity as ce
            ce = tf.add(ce_0, ce_1)
            ce_avg = tf.reduce_mean(ce)
            rescaled_factor = tf.divide(ce_avg, f1_avg + epsilon)
            f1_avg = tf.multiply(rescaled_factor, f1_avg)
        
        return f1_avg
    
    return focal_loss_fixed


# In[52]:


def f1_loss(y_true, y_pred):
    # y_true:真实标签0或者1；y_pred:为正类的概率
    loss = 2 * tf.reduce_sum(y_true * y_pred) / tf.reduce_sum(y_true + y_pred) + K.epsilon()
    return -loss


# In[53]:


def transform_y(y_true, y_pred):
    mask_value = tf.constant(-1)
    mask_y_true = tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.cast(mask_value, dtype=tf.int32))
#     print(f"mask_y_true:{mask_y_true}")
#     y_true_ = tf.cond(tf.equal(y_true, mask_value), lambda: 0, lambda: y_true)
    y_true_ = tf.cast(y_true, dtype=tf.int32) * tf.cast(mask_y_true, dtype=tf.int32)
    y_pred_ = tf.cast(y_pred, dtype=tf.float32) * tf.cast(mask_y_true, dtype=tf.float32)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")
    
    return tf.cast(y_true_, dtype=tf.float32), tf.cast(y_pred_, dtype=tf.float32)


# In[54]:


def my_binary_crossentropy(y_true, y_pred, class_weight_0, class_weight_1):
#     print(f"y_true: {y_true}")
    mask_value = tf.constant(-1)
#     mask_y_true = tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.cast(mask_value, dtype=tf.int32))
    
#     mask = tf.zeros(shape=y_true.shape)
    zero_value = tf.constant(0)
#     print(f"cond0: {tf.equal(y_true, mask_value)}")
#     print(f"cond1: {tf.equal(y_true, zero_value)}")
#     weight = [tf.cond(tf.equal(x, mask_value), lambda: 0, tf.cond(tf.equal(x, zero_value), lambda: class_weights[0], lambda: class_weights[1])) for x in y_true]
#     weight = [0 if x[0]==-1 else class_weights[x[0]] for x in y_true]
    y_true_0 = tf.equal(tf.cast(y_true, dtype=tf.int32), tf.cast(tf.constant(0), dtype=tf.int32))
    weight_0 = tf.cast(y_true_0, dtype=tf.float32) * tf.cast(tf.constant(class_weight_0), dtype=tf.float32)
    y_true_1 = tf.equal(tf.cast(y_true, dtype=tf.int32), tf.cast(tf.constant(1), dtype=tf.int32))
    weight_1 = tf.cast(y_true_1, dtype=tf.float32) * tf.cast(tf.constant(class_weight_1), dtype=tf.float32)
    weight = weight_0 + weight_1
#     print(f"weight: {weight}")
    
    bin_loss = binary_crossentropy(y_true, y_pred)
#     print(f"bin_loss: {bin_loss}")
#     f1_loss = f1_loss(y_true, y_pred)
#     loss = bin_loss + f1_loss
    
    loss_ = tf.cast(bin_loss, dtype=tf.float32) * tf.cast(weight, dtype=tf.float32)
#     print(f"loss_: {loss_}")
    loss_abs = tf.abs(loss_)
#     print(f"loss_abs: {loss_abs}")

    return loss_abs


# In[55]:


def my_binary_crossentropy_A(y_true, y_pred):
    return my_binary_crossentropy(y_true, y_pred, labelA_class_weights[0], labelA_class_weights[1])

def my_binary_crossentropy_B(y_true, y_pred):
    return my_binary_crossentropy(y_true, y_pred, labelB_class_weights[0], labelB_class_weights[1])


# In[56]:


def tarnsform_metrics(y_true, y_pred):
    y_true_, y_pred_ = y_true.numpy(), y_pred.numpy()
    for i in range(y_true_.shape[0]):
        for j in range(y_true_.shape[1]):
            if y_true_[i][j] == -1:
                y_true_[i][j] = 0
                y_pred_[i][j] = random.choice([0, 1])
            if y_pred_[i][j] > 0.5:
                y_pred_[i][j] = 1
            else:
                y_pred_[i][j] = 0
    return y_true_, y_pred_


# In[57]:


def my_binary_accuracy(y_true, y_pred):
#     print("my_binary_accuracy")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    accuracy = binary_accuracy(y_true_, y_pred_)
    return accuracy


# In[58]:


def my_f1_score(y_true, y_pred):
#     print("my_f1_score")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    return f1_score(y_true_, y_pred_, average='macro')


# In[59]:


def get_model():
#     K.clear_session()

    bert_model = TFBertModel.from_pretrained(bert_path, from_pt=True, trainable=True)
    for l in bert_model.layers:
        l.trainable = True

    input_ids_texta = Input(shape=(None,), dtype='int32', name='input_ids_texta')
    input_token_type_ids_texta = Input(shape=(None,), dtype='int32', name='input_token_type_ids_texta')
    input_attention_mask_texta = Input(shape=(None,), dtype='int32', name='input_attention_mask_texta')
    input_ids_textb = Input(shape=(None,), dtype='int32', name='input_ids_textb')
    input_token_type_ids_textb = Input(shape=(None,), dtype='int32', name='input_token_type_ids_textb')
    input_attention_mask_textb = Input(shape=(None,), dtype='int32', name='input_attention_mask_textb')
    input_bm25 = Input(shape=(1), dtype='float32', name='input_bm25')
    input_tf_cosine = Input(shape=(1), dtype='float32', name='input_tf_cosine')
    input_tfidf_cosine = Input(shape=(1), dtype='float32', name='input_tfidf_cosine')
    input_cat_texta = Input(shape=(1), dtype='float32', name='input_cat_texta')
    input_cat_textb = Input(shape=(1), dtype='float32', name='input_cat_textb')

    bert_output_texta = bert_model({'input_ids':input_ids_texta, 'token_type_ids':input_token_type_ids_texta, 'attention_mask':input_attention_mask_texta}, return_dict=False, training=True)
    projection_logits_texta = bert_output_texta[0]
    bert_cls_texta = Lambda(lambda x: x[:, 0])(projection_logits_texta) # 取出[CLS]对应的向量用来做分类

    bert_output_textb = bert_model({'input_ids':input_ids_textb, 'token_type_ids':input_token_type_ids_textb, 'attention_mask':input_attention_mask_textb}, return_dict=False, training=True)
    projection_logits_textb = bert_output_textb[0]
    bert_cls_textb = Lambda(lambda x: x[:, 0])(projection_logits_textb) # 取出[CLS]对应的向量用来做分类

    subtracted = Subtract()([bert_cls_texta, bert_cls_textb])
    cos = Dot(axes=1, normalize=True)([bert_cls_texta, bert_cls_textb]) # dot=1按行点积，normalize=True输出余弦相似度

    bert_cls = concatenate([bert_cls_texta, bert_cls_textb, subtracted, cos, input_bm25, input_tf_cosine, input_tfidf_cosine, input_cat_texta, input_cat_textb], axis=-1)

    dense_A_0 = Dense(256, activation='relu')(bert_cls)
    dropout_A_0 = Dropout(0.2)(dense_A_0)
    dense_A_1 = Dense(32, activation='relu')(dropout_A_0)
    dropout_A_1 = Dropout(0.2)(dense_A_1)
    output_A = Dense(1, activation='sigmoid', name='output_A')(dropout_A_1)

    dense_B_0 = Dense(256, activation='relu')(bert_cls)
    dropout_B_0 = Dropout(0.2)(dense_B_0)
    dense_B_1 = Dense(32, activation='relu')(dropout_B_0)
    dropout_B_1 = Dropout(0.2)(dense_B_1)
    output_B = Dense(1, activation='sigmoid', name='output_B')(dropout_B_1)

    input_data = {
        'ids_texta':input_ids_texta,
        'token_type_ids_texta':input_token_type_ids_texta,
        'attention_mask_texta':input_attention_mask_texta,
        'ids_textb':input_ids_textb,
        'token_type_ids_textb':input_token_type_ids_textb,
        'attention_mask_textb':input_attention_mask_textb,
        'bm25':input_bm25,
        'tf_cosine':input_tf_cosine,
        'tfidf_cosine':input_tfidf_cosine,
        'cat_texta':input_cat_texta,
        'cat_textb':input_cat_textb,
    }
    output_data = {
        'labelA':output_A,
        'labelB':output_B,
    }
    model = Model(input_data, output_data)
    model.compile(
#                   loss=my_binary_crossentropy,
#                   loss={
#                       'output_A':my_binary_crossentropy_A,
#                       'output_B':my_binary_crossentropy_B,
#                   },
                  loss={
                      'labelA':my_binary_crossentropy_A,
                      'labelB':my_binary_crossentropy_B,
                  },
#                   loss='binary_crossentropy',
#                   loss=binary_crossentropy,
                  optimizer=Adam(1e-5),    #用足够小的学习率
#                   metrics=[my_binary_accuracy, my_f1_score]
                  metrics='accuracy'
                 )
#     print(model.summary())
    return model


# In[60]:


early_stopping = EarlyStopping(monitor='loss', patience=3)   #早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="loss", verbose=1, factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint(check_point_path, monitor='loss', verbose=2, save_best_only=True, save_weights_only=True) #保存最好的模型
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}", update_freq=50)


# ## 模型训练

# In[61]:


def get_step(sample_count, batch_size):
    step = sample_count // batch_size
    if sample_count % batch_size != 0:
        step += 1
    return step


# In[62]:


# model = get_model()
# plot_model(model, "keras_bert_transformers_two_text_input_SubStract_bm25cosine_1.png", show_shapes=True)


# In[63]:


# model.load_weights(check_point_path)


# In[64]:


# batch_size = 2
# epochs = 10

# train_dataset_iterator = batch_iter(data_path, train_file_name, tokenizer, batch_size)
# train_step = get_step(sample_count, batch_size)

# model.fit(
#     train_dataset_iterator,
#     # steps_per_epoch=10,
#     steps_per_epoch=train_step,
#     epochs=epochs,
# #       validation_data=dev_dataset_iterator,
#   # validation_steps=2,
# #       validation_steps=dev_step,
# #     validation_split=0.2,
# #     class_weight={
# #         'output_A':labelA_class_weights,
# #         'output_B':labelB_class_weights,
# #     },
#     callbacks=[early_stopping, plateau, checkpoint, tensorboard_callback],
#     verbose=1
# )

# model.save_weights(weights_path)
# # model_json = model.to_json()
# # with open(config_path, 'w', encoding='utf-8') as file:
# #     file.write(model_json)

# save_test_result(model, result_path)


# In[65]:


# model = get_model()
# # with open(config_path, 'r', encoding='utf-8') as json_file:
# #     loaded_model_json = json_file.read()
# # model = model_from_json(loaded_model_json)
# model.load_weights(check_point_path)
# save_test_result(model, "trained_model_substract_1/multi_keras_bert_sohu_test_result_epoch6.csv")


# In[66]:


batch_size = 2 * strategy.num_replicas_in_sync
n_epochs = 10
start_epoch = 1

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_dataset_iterator = get_dataset("data/shuffle_total_file.tfrecord", batch_size=batch_size, options=options)
train_step = get_step(train_dataset_len, batch_size)

with strategy.scope():
    model = get_model()
    print(model.summary())
    plot_model(model, "keras_bert_transformers_two_text_input_SubStract_bm25cosine_1.png", show_shapes=True)
    
    model.load_weights(check_point_path)

for epoch in range(start_epoch, start_epoch + n_epochs):
    model.fit(
        train_dataset_iterator,
        steps_per_epoch=train_step,
        epochs=1,
        callbacks=[early_stopping, plateau, checkpoint, tensorboard_callback],
        verbose=1
    )

    model.save_weights(weights_path)

    save_test_result(weights_path, f"{result_path}.epoch_{epoch}.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 模型加载及测试

# ## load_weights

# ## load_model

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




