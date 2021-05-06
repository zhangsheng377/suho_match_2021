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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import re
from collections import Counter
import random
import json

from tqdm import tqdm
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
from keras.metrics import top_k_categorical_accuracy, binary_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import SparseCategoricalCrossentropy, binary_crossentropy
from transformers import (
    BertTokenizer,
    TFBertForPreTraining,
    TFBertModel,
)
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
import torch


# In[6]:


tf.__version__


# In[7]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[8]:


# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()


# In[ ]:





# In[9]:


data_path = "sohu2021_open_data_clean/"
text_max_length = 512
bert_path = r"chinese_L-12_H-768_A-12"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 构建标签表

# In[10]:


label_to_id = {'0':0, '1':1}


# In[11]:


labels = [0, 1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 构建原数据文本迭代器

# In[12]:


def _transform_text(text):
   text = text.strip().replace('\n', '。').replace('\t', '').replace('\u3000', '')
   return re.sub(r'。+', '。', text)


# In[13]:


def get_data_iterator(data_path, file_name):
    # TODO: 随机取
    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        if not os.path.isdir(category_path):
            continue
            
        file_path = os.path.join(category_path, file_name)
        if not os.path.isfile(file_path):
            continue
        
#         print(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                data['source'] = _transform_text(data['source'])
                if len(data['source']) == 0:
                    print('source:', line, data)
                    break
#                     continue
                    
                data['target'] = _transform_text(data['target'])
                if len(data['target']) == 0:
                    print('target:', line, data)
                    break
#                     continue
                
                label_name_list = list(key for key in data.keys() if key[:5]=='label')
                if len(label_name_list) != 1:
                    print('label_name_list:', line, data)
                    break
#                     continue
                label_name = label_name_list[0]
                if data[label_name] not in label_to_id.keys():
                    print('label_name:', line, data, label_name)
                    break
#                     continue
                    
                yield data['source'], data['target'], label_to_id[data[label_name]]


# In[14]:


it = get_data_iterator(data_path, "train.txt")


# In[15]:


next(it)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 获取数据集样本个数

# In[16]:


def get_sample_num(data_path, file_name):
    count = 0
    it = get_data_iterator(data_path, file_name)
    for data in tqdm(it):
        count += 1
    return count


# In[17]:


train_sample_count = get_sample_num(data_path, "train.txt")


# In[18]:


dev_sample_count = get_sample_num(data_path, "valid.txt")


# In[19]:


train_sample_count, dev_sample_count


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 构建数据迭代器

# In[20]:


tokenizer = BertTokenizer.from_pretrained(bert_path)


# In[21]:


def _get_indices(text, text_pair=None):
    return tokenizer.encode(text=text,
                            text_pair=text_pair,
                            max_length=text_max_length, 
                            add_special_tokens=True, 
                            padding='max_length', 
                            truncation_strategy='only_first', 
#                                          return_tensors='tf'
                            )


# In[22]:


def get_keras_bert_iterator(data_path, file_name, tokenizer):
    while True:
        data_it = get_data_iterator(data_path, file_name)
        for source, target, label in data_it:
            indices = _get_indices(text=source, 
                                   text_pair=target)
            yield indices, label


# In[23]:


it = get_keras_bert_iterator(data_path, "train.txt", tokenizer)


# In[24]:


# next(it)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 构建批次数据迭代器

# In[25]:


def batch_iter(data_path, file_name, tokenizer, batch_size=64, shuffle=True):
    """生成批次数据"""
    keras_bert_iter = get_keras_bert_iterator(data_path, file_name, tokenizer)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(keras_bert_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        indices_list = []
        label_list = []
        for data in data_list:
            indices, label = data
            indices_list.append(indices)
            label_list.append(label)

        yield np.array(indices_list), np.array(label_list)


# In[26]:


it = batch_iter(data_path, "train.txt", tokenizer, batch_size=1)


# In[27]:


# next(it)


# In[28]:


it = batch_iter(data_path, "train.txt", tokenizer, batch_size=2)


# In[29]:


next(it)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 定义base模型

# In[30]:


# !transformers-cli convert --model_type bert \
#   --tf_checkpoint chinese_L-12_H-768_A-12/bert_model.ckpt \
#   --config chinese_L-12_H-768_A-12/bert_config.json \
#   --pytorch_dump_output chinese_L-12_H-768_A-12/pytorch_model.bin


# In[31]:


# bert_model = TFBertForPreTraining.from_pretrained("./chinese_L-12_H-768_A-12/", from_pt=True)


# In[32]:


# # it = get_keras_bert_iterator(r"data/keras_bert_train.txt", cat_to_id, tokenizer)
# it = batch_iter(r"data/keras_bert_train.txt", cat_to_id, tokenizer, batch_size=1)
# out = bert_model(next(it)[0])
# out[0]


# In[33]:


def get_model(label_list):
    K.clear_session()
    
    bert_model = TFBertForPreTraining.from_pretrained(bert_path, from_pt=True)
 
    input_indices = Input(shape=(None,), dtype='int32')
 
    bert_output = bert_model(input_indices)
    projection_logits = bert_output[0]
    bert_cls = Lambda(lambda x: x[:, 0])(projection_logits) # 取出[CLS]对应的向量用来做分类
    
    dropout = Dropout(0.5)(bert_cls)
    output = Dense(len(label_list), activation='softmax')(dropout)
 
    model = Model(input_indices, output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=['accuracy'])
    print(model.summary())
    return model


# In[34]:


early_stopping = EarlyStopping(monitor='val_loss', patience=3)   #早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="val_accuracy", verbose=1, mode='max', factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint('trained_model/keras_bert_sohu.hdf5', monitor='val_loss',verbose=2, save_best_only=True, mode='max', save_weights_only=True) #保存最好的模型


# ## 模型训练

# In[35]:


def get_step(sample_count, batch_size):
    step = sample_count // batch_size
    if sample_count % batch_size != 0:
        step += 1
    return step


# In[36]:


# batch_size = 2
# train_step = get_step(train_sample_count, batch_size)
# dev_step = get_step(dev_sample_count, batch_size)

# train_dataset_iterator = batch_iter(data_path, "train.txt", tokenizer, batch_size)
# dev_dataset_iterator = batch_iter(data_path, "valid.txt", tokenizer, batch_size)

# model = get_model(labels)

# #模型训练
# model.fit(
#     train_dataset_iterator,
#     steps_per_epoch=10,
# #     steps_per_epoch=train_step,
#     epochs=5,
#     validation_data=dev_dataset_iterator,
#     validation_steps=2,
# #     validation_steps=dev_step,
#     callbacks=[early_stopping, plateau, checkpoint],
#     verbose=1
# )

# model.save_weights("trained_model/keras_bert_sohu_final.weights")
# model.save("trained_model/keras_bert_sohu_final.model")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 多任务分支模型

# ## 构建数据迭代器

# In[37]:


label_type_to_id = {'labelA':0, 'labelB':1}


# In[38]:


def get_text_iterator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line


# In[39]:


def get_data_iterator(data_path, file_names):
    # TODO: 随机取
    file_iters = []
    for file_name in file_names:
      for category in os.listdir(data_path):
          category_path = os.path.join(data_path, category)
          if not os.path.isdir(category_path):
              continue
              
          file_path = os.path.join(category_path, file_name)
          if not os.path.isfile(file_path):
              continue
              
          
          file_iter = get_text_iterator(file_path)
          cat_source = 0
          if category[0] == '长':
            cat_source = 1
          cat_target = 0
          if category[1] == '长':
            cat_target = 1
          file_iters.append((file_iter, cat_source, cat_target))
        
    while len(file_iters) > 0:
        i = random.randrange(len(file_iters))
        line = next(file_iters[i][0], None)
        cat_source = file_iters[i][1]
        cat_target = file_iters[i][2]
        if line is None:
            del file_iters[i]
            continue
            
        data = json.loads(line)

        data['source'] = _transform_text(data['source'])
        if len(data['source']) == 0:
            print('source:', line, data)
            break
#                     continue

        data['target'] = _transform_text(data['target'])
        if len(data['target']) == 0:
            print('target:', line, data)
            break
#                     continue

        label_name_list = list(key for key in data.keys() if key[:5]=='label')
        if len(label_name_list) != 1:
            print('label_name_list:', line, data)
            break
#                     continue
        label_name = label_name_list[0]
        if data[label_name] not in label_to_id.keys():
            print('label_name:', line, data, label_name)
            break
#                     continue
        
        label_dict = {key: -1 for key in label_type_to_id.keys()}
        label_dict[label_name] = label_to_id[data[label_name]]
        if label_dict['labelA'] == 0:
            label_dict['labelB'] = 0
        if label_dict['labelB'] == 1:
            label_dict['labelA'] = 1

        label_dict['labelC'] = 0
        if label_dict['labelA'] == 1 or label_dict['labelB'] == 1:
          label_dict['labelC'] = 1

        yield data['source'], data['target'], cat_source, cat_target, label_dict['labelA'], label_dict['labelB'], label_dict['labelC']


# In[45]:


it = get_data_iterator(data_path, ["train.txt", "valid.txt", "round2.txt"])


# In[46]:


next(it)


# In[47]:


sample_count = get_sample_num(data_path, ["train.txt", "valid.txt", "round2.txt"])
sample_count


# In[48]:


def get_sample_y(data_path, file_names):
    labelA_list = []
    labelB_list = []
    labelC_list = []
    it = get_data_iterator(data_path, file_names)
    for source, target, cat_source, cat_target, labelA, labelB, labelC in tqdm(it):
        if labelA != -1:
          labelA_list.append(labelA)
        if labelB != -1:
          labelB_list.append(labelB)
        labelC_list.append(labelC)
    return labelA_list, labelB_list, labelC_list


# In[49]:


# labelA_list, labelB_list, labelC_list = get_sample_y(data_path, ["train.txt", "valid.txt"])
# labelA_class_weights = class_weight.compute_class_weight('balanced', np.unique(labelA_list), labelA_list)
# labelB_class_weights = class_weight.compute_class_weight('balanced', np.unique(labelB_list), labelB_list)
# labelC_class_weights = class_weight.compute_class_weight('balanced', np.unique(labelC_list), labelC_list)
# labelA_class_weights, labelB_class_weights, labelC_class_weights


# In[50]:


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


# In[51]:


def get_keras_bert_iterator(data_path, file_names, tokenizer):
    while True:
        data_it = get_data_iterator(data_path, file_names)
        for source, target, cat_source, cat_target, labelA, labelB, labelC in data_it:
            data_source = _get_indices(text=source)
            data_target = _get_indices(text=target)
#             print(indices, type(indices), len(indices))
            yield data_source['input_ids'], data_source['token_type_ids'], data_source['attention_mask'], data_target['input_ids'], data_target['token_type_ids'], data_target['attention_mask'], cat_source, cat_target, labelA, labelB, labelC


# In[53]:


it = get_keras_bert_iterator(data_path, ["train.txt", "valid.txt", "round2.txt"], tokenizer)


# In[54]:


# next(it)


# In[55]:


def batch_iter(data_path, file_names, tokenizer, batch_size=64, shuffle=True):
    """生成批次数据"""
    keras_bert_iter = get_keras_bert_iterator(data_path, file_names, tokenizer)
    while True:
        data_list = []
        for _ in range(batch_size):
            data = next(keras_bert_iter)
            data_list.append(data)
        if shuffle:
            random.shuffle(data_list)
        
        input_ids_texta_list = []
        token_type_ids_texta_list = []
        attention_mask_texta_list = []
        input_ids_textb_list = []
        token_type_ids_textb_list = []
        attention_mask_textb_list = []
        cat_texta_list = []
        cat_textb_list = []
        labelA_list = []
        labelB_list = []
        labelC_list = []
        for data in data_list:
            input_ids_texta, token_type_ids_texta, attention_mask_texta, input_ids_textb, token_type_ids_textb, attention_mask_textb, cat_texta, cat_textb, labelA, labelB, labelC = data
#             print(indices, type(indices))
            input_ids_texta_list.append(input_ids_texta)
            token_type_ids_texta_list.append(token_type_ids_texta)
            attention_mask_texta_list.append(attention_mask_texta)
            input_ids_textb_list.append(input_ids_textb)
            token_type_ids_textb_list.append(token_type_ids_textb)
            attention_mask_textb_list.append(attention_mask_textb)
            cat_texta_list.append(cat_texta)
            cat_textb_list.append(cat_textb)
            labelA_list.append(labelA)
            labelB_list.append(labelB)
            labelC_list.append(labelC)

        yield [np.array(input_ids_texta_list), np.array(token_type_ids_texta_list), np.array(attention_mask_texta_list), 
               np.array(input_ids_textb_list), np.array(token_type_ids_textb_list), np.array(attention_mask_textb_list), 
               np.array(cat_texta_list), np.array(cat_textb_list)], \
            [np.array(labelA_list, dtype=np.int32), np.array(labelB_list, dtype=np.int32), np.array(labelC_list, dtype=np.int32)]


# In[56]:


it = batch_iter(data_path, ["train.txt", "valid.txt", "round2.txt"], tokenizer, batch_size=2)


# In[57]:


next(it)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


def get_test_data_iterator(data_path, file_name):
  # print(data_path)
  for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)
    # print(category_path)
    if not os.path.isdir(category_path):
      # print(f"{category_path} not dir")
      continue
        
    file_path = os.path.join(category_path, file_name)
    # print(file_path)
    if not os.path.isfile(file_path):
      # print(f"{file_path} not file")
      continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
      for line in f:
        # print(line)
        data = json.loads(line)
        
        data['source'] = _transform_text(data['source'])
        if len(data['source']) == 0:
          print('source:', line, data)
          break
            
        data['target'] = _transform_text(data['target'])
        if len(data['target']) == 0:
          print('target:', line, data)
          break

        cat_source = 0
        if category[0] == '长':
          cat_source = 1
        cat_target = 0
        if category[1] == '长':
          cat_target = 1
            
        yield data['source'], data['target'], cat_source, cat_target, data['id']


# In[59]:


def get_test_keras_bert_iterator(data_path, file_name):
  it = get_test_data_iterator(data_path, file_name)
  for source, target, cat_source, cat_target, id in it:
    data_source = _get_indices(text=source)
    data_target = _get_indices(text=target)
    yield data_source['input_ids'], data_source['token_type_ids'], data_source['attention_mask'], data_target['input_ids'], data_target['token_type_ids'], data_target['attention_mask'], cat_source, cat_target, id


# In[60]:


def get_test_iterator(data_path, file_name):
  it = get_test_keras_bert_iterator(data_path, file_name)
  for input_ids_texta, token_type_ids_texta, attention_mask_texta, input_ids_textb, token_type_ids_textb, attention_mask_textb, cat_source, cat_target, id in it:
    yield [np.array([input_ids_texta]), np.array([token_type_ids_texta]), np.array([attention_mask_texta]), np.array([input_ids_textb]), np.array([token_type_ids_textb]), np.array([attention_mask_textb]), np.array([cat_source]), np.array([cat_target])], id


# In[61]:


it = get_test_iterator(data_path, "test_with_id.txt")
# next(it)


# In[62]:


def save_test_result(model, result_file):
  it = get_test_iterator(data_path, "test_with_id.txt")
  print("      ", end="")
  count = 0
  with open(result_file, 'w', encoding='utf-8') as f:
    for data, id in it:
      predict = model.predict(data)
      if id[-1] == 'a':
        predict_cls = 1 if predict[0][0][0] > 0.5 else 0
      elif id[-1] == 'b':
        predict_cls = 1 if predict[1][0][0] > 0.5 else 0
      else:
        print(id)
        continue
      f.write(f"{id},{predict_cls}\n")
      count += 1
      print(f"\b\b\b\b\b\b{count}", end="")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 定义模型

# In[63]:


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


# In[64]:


def transform_y(y_true, y_pred):
    mask_value = tf.constant(-1)
    mask_y_true = tf.not_equal(tf.cast(y_true, dtype=tf.int32), tf.cast(mask_value, dtype=tf.int32))
#     print(f"mask_y_true:{mask_y_true}")
#     y_true_ = tf.cond(tf.equal(y_true, mask_value), lambda: 0, lambda: y_true)
    y_true_ = tf.cast(y_true, dtype=tf.int32) * tf.cast(mask_y_true, dtype=tf.int32)
    y_pred_ = tf.cast(y_pred, dtype=tf.float32) * tf.cast(mask_y_true, dtype=tf.float32)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")
    
    return tf.cast(y_true_, dtype=tf.float32), tf.cast(y_pred_, dtype=tf.float32)


# In[65]:


def my_binary_crossentropy(y_true, y_pred):
    # print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true, y_pred = transform_y(y_true, y_pred)
    # print(f"y_true_:{y_true}, y_pred_:{y_pred}")

    # loss = binary_crossentropy(y_true, y_pred)
    loss = variant_focal_loss()(y_true, y_pred)

    # y_true0 = tf.constant([y_true.numpy()[0]])
    # y_true1 = tf.constant([y_true.numpy()[1]])
    # y_pred0 = tf.constant([y_pred.numpy()[0]])
    # y_pred1 = tf.constant([y_pred.numpy()[1]])
    # loss0 = variant_focal_loss()(y_true0, y_pred0)
    # loss1 = variant_focal_loss()(y_true1, y_pred1)
    # print(f"y_true_0:{y_true0}, y_pred_0:{y_pred0}")
    # print(f"y_true_1:{y_true1}, y_pred_1:{y_pred1}")
    # print(f"loss0:{loss0}")
    # print(f"loss1:{loss1}")

    # print(f"loss:{loss}")
    return loss


# In[66]:


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


# In[67]:


def my_binary_accuracy(y_true, y_pred):
#     print("my_binary_accuracy")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    accuracy = binary_accuracy(y_true_, y_pred_)
    return accuracy


# In[68]:


def my_f1_score(y_true, y_pred):
#     print("my_f1_score")
#     print(f"y_true:{y_true}, y_pred:{y_pred}")
    
    y_true_, y_pred_ = tarnsform_metrics(y_true, y_pred)
#     print(f"y_true_:{y_true_}, y_pred_:{y_pred_}")

    return f1_score(y_true_, y_pred_, average='macro')


# In[69]:


def get_model():
    K.clear_session()
    
    bert_model = TFBertModel.from_pretrained(bert_path, from_pt=True, trainable=True)
 
    input_ids_texta = Input(shape=(None,), dtype='int32')
    input_token_type_ids_texta = Input(shape=(None,), dtype='int32')
    input_attention_mask_texta = Input(shape=(None,), dtype='int32')
    input_ids_textb = Input(shape=(None,), dtype='int32')
    input_token_type_ids_textb = Input(shape=(None,), dtype='int32')
    input_attention_mask_textb = Input(shape=(None,), dtype='int32')
    input_token_type_ids_textb = Input(shape=(None,), dtype='int32')
    input_cat_texta = Input(shape=(1), dtype='float32')
    input_cat_textb = Input(shape=(1), dtype='float32')
 
    bert_output_texta = bert_model({'input_ids':input_ids_texta, 'token_type_ids':input_token_type_ids_texta, 'attention_mask':input_attention_mask_texta}, return_dict=False, training=True)
    projection_logits_texta = bert_output_texta[0]
    bert_cls_texta = Lambda(lambda x: x[:, 0])(projection_logits_texta) # 取出[CLS]对应的向量用来做分类

    bert_output_textb = bert_model({'input_ids':input_ids_textb, 'token_type_ids':input_token_type_ids_textb, 'attention_mask':input_attention_mask_textb}, return_dict=False, training=True)
    projection_logits_textb = bert_output_textb[0]
    bert_cls_textb = Lambda(lambda x: x[:, 0])(projection_logits_textb) # 取出[CLS]对应的向量用来做分类

    bert_cls = concatenate([bert_cls_texta, bert_cls_textb, input_cat_texta, input_cat_textb], axis=-1)
    
    dropout_A = Dropout(0.5)(bert_cls)
    output_A = Dense(1, activation='sigmoid')(dropout_A)
    
    dropout_B = Dropout(0.5)(bert_cls)
    output_B = Dense(1, activation='sigmoid')(dropout_B)

    dropout_C = Dropout(0.5)(bert_cls)
    output_C = Dense(1, activation='sigmoid')(dropout_C)
 
    model = Model([input_ids_texta, input_token_type_ids_texta, input_attention_mask_texta, input_ids_textb, input_token_type_ids_textb, input_attention_mask_textb, input_cat_texta, input_cat_textb], [output_A, output_B, output_C])
    model.compile(
                  loss=[my_binary_crossentropy, my_binary_crossentropy, 'binary_crossentropy'],
#                   loss='binary_crossentropy',
#                   loss=binary_crossentropy,
                  optimizer=Adam(1e-5),    #用足够小的学习率
                  metrics=[my_binary_accuracy, my_f1_score]
#                   metrics='accuracy'
                 )
    print(model.summary())
    return model


# In[70]:


early_stopping = EarlyStopping(monitor='loss', patience=3)   #早停法，防止过拟合
plateau = ReduceLROnPlateau(monitor="loss", verbose=1, mode='max', factor=0.5, patience=2) #当评价指标不在提升时，减少学习率
checkpoint = ModelCheckpoint('trained_model/multi_keras_bert_sohu.weights', monitor='loss', verbose=2, save_best_only=True, save_weights_only=True) #保存最好的模型
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq=5000)


# ## 模型训练

# In[71]:


# batch_size = 2
# train_step = get_step(train_sample_count, batch_size)
# dev_step = get_step(dev_sample_count, batch_size)

# train_dataset_iterator = batch_iter(data_path, "train.txt", tokenizer, batch_size)
# dev_dataset_iterator = batch_iter(data_path, "valid.txt", tokenizer, batch_size)

# model = get_model()

# epoch = 2
# #模型训练
# for i in range(epoch):
#   model.fit(
#       train_dataset_iterator,
#       # steps_per_epoch=10,
#       steps_per_epoch=train_step,
#       epochs=1,
#       validation_data=dev_dataset_iterator,
#       # validation_steps=2,
#       validation_steps=dev_step,
#       # callbacks=[early_stopping, plateau, checkpoint],
#       verbose=1
#   )

#   model.save_weights(f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_epoch{i}.weights")
#   model.save(f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_epoch{i}.model")

#   save_test_result(model, f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_test_result_epoch{i}.csv")


# In[72]:


batch_size = 2

train_dataset_iterator = batch_iter(data_path, ["train.txt", "valid.txt", "round2.txt"], tokenizer, batch_size)
train_step = get_step(sample_count, batch_size)

model = get_model()

model.fit(
  train_dataset_iterator,
  # steps_per_epoch=10,
  steps_per_epoch=train_step,
  epochs=5,
#       validation_data=dev_dataset_iterator,
  # validation_steps=2,
#       validation_steps=dev_step,
  callbacks=[early_stopping, plateau, checkpoint, tensorboard_callback],
  verbose=1
)

model.save_weights(f"trained_model/multi_keras_bert_sohu_final.weights")

save_test_result(model, f"trained_model/multi_keras_bert_sohu_test_result_final.csv")


# In[ ]:


# batch_size = 2

# train_dataset_iterator = batch_iter(data_path, ["train.txt", "valid.txt"], tokenizer, batch_size)

# model = get_model()


# In[ ]:


# model.load_weights(f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_step.weights")


# In[ ]:


# save_test_result(model, f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_test_result_.csv")


# In[ ]:


# epoch = 1
# per_train_step = 5000
# per_epoch_step = (sample_count // batch_size + 1) // per_train_step + 1

# # start_i = (3+8+1+2) % per_epoch_step
# start_i = (0+1) % per_epoch_step
# for i in range(start_i):
#   for j in range(per_train_step):
#     next(train_dataset_iterator)


# #模型训练
# for epoch_ in range(epoch):
#   for i in range(start_i, per_epoch_step):
#     model.fit(
#         train_dataset_iterator,
#         steps_per_epoch=per_train_step,
#         epochs=1,
#         # class_weight=[labelA_class_weights, labelB_class_weights, labelC_class_weights],
#         verbose=1
#     )

#     model.save_weights(f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_step.weights")
#     # model.save(f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_step.model")

#   save_test_result(model, f"/content/drive/MyDrive/sohu_match/multi_keras_bert_sohu_test_result_epoch.csv")


# In[ ]:


# dev_dataset_iterator = batch_iter(data_path, "valid.txt", tokenizer, batch_size=1)
# data = next(dev_dataset_iterator)
# model.predict(data[0]), data[1]


# In[ ]:


# data = next(dev_dataset_iterator)
# model.predict(data[0]), data[1]


# In[ ]:


# model.save_weights("/content/drive/MyDrive/multi_keras_bert_sohu_final.weights")
# model.save("/content/drive/MyDrive/multi_keras_bert_sohu_final.model")


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




