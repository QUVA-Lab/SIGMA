# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
from typing import Dict
import json
import numpy as np
import math


# change seed to pick up  a different subset of  random samples
seed = 99999
#seed = 12345
#seed = 19

def get_subset_data(video_names,annotations,num_of_examples):

    examples_per_class =   int(math.ceil(num_of_examples / len(set(annotations))))  
    print("original data length", len(video_names) )
    print("subset data", examples_per_class,num_of_examples)

    random_state = np.random.RandomState(seed)

    annotations = np.array(annotations)
    video_names = np.array(video_names )

    subset_video_names = []
    subset_annotations = []
    for class_label in set(annotations): # sample uniformly from each class

          subset_indexes = np.where(annotations == class_label)
          #print("shape",subset_indexes[0].shape[0])

          if examples_per_class > subset_indexes[0].shape[0]:

                           ran_indicies = np.array(random_state.choice(subset_indexes[0].shape[0],subset_indexes[0].shape[0],replace=False))
          else:
                           ran_indicies = np.array(random_state.choice(subset_indexes[0].shape[0],examples_per_class,replace=False))

          indicies_100 = (subset_indexes[0][ran_indicies])
          temp_annotations =annotations[indicies_100]
          temp_names=  video_names[indicies_100]
          subset_video_names.extend(temp_names)
          subset_annotations.extend(temp_annotations)
    video_names  = list(subset_video_names)
    annotations = list(subset_annotations)
    print(len(video_names),len(annotations))
    #print(video_names)
    #print(annotations)
    return video_names, annotations



def get_filenames_and_labels_ucf(data_root,subset,num_of_examples=0):

   DATA_PATH = data_root + 'UCF-101/'
   ANNO_PATH = data_root + 'ucfTrainTestlist'

   filenames = []
   labels = []
   classes_fn = f'{ANNO_PATH}/classInd.txt'
   classes = [l.strip().split()[1] for l in open(classes_fn)]

   if 'train' in subset:
             filenames = [ln.strip().split()[0] for ln in open(f'{ANNO_PATH}/trainlist01.txt')]
             labels = [fn.split('/')[0] for fn in filenames]
             labels = [classes.index(cls) for cls in labels]
             filenames = [DATA_PATH+ name for name in filenames ]
   else:
             filenames = [ln.strip().split()[0] for ln in open(f'{ANNO_PATH}/testlist01.txt')]
             labels = [fn.split('/')[0] for fn in filenames]
             labels = [classes.index(cls) for cls in labels]
             filenames = [DATA_PATH+ name for name in filenames ]

   if 'train' in subset and num_of_examples!=0:
                  filenames, labels = get_subset_data(filenames,labels,num_of_examples)

   return filenames,labels

def get_filenames_and_labels_gym(data_root,subset,num_of_examples=0):

   DATA_PATH = data_root + 'subactions/'
   ANNO_PATH = data_root + 'annotations'

   filenames = []
   labels = []

   if 'train' in subset:

          for ln in open(f'{ANNO_PATH}/gym99_train.txt'):
                 file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                 if os.path.isfile(DATA_PATH+'/'+file_name):
                         #print(file_name,label)
                         filenames.append(DATA_PATH+'/'+file_name)
                         labels.append(label)

   else:
          for ln in open(f'{ANNO_PATH}/gym99_val.txt'):
                 file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                 if os.path.isfile(DATA_PATH+'/'+file_name):
                         filenames.append(DATA_PATH+'/'+file_name)
                         labels.append(label)
   if 'train' in subset and num_of_examples!=0:
                  filenames, labels = get_subset_data(filenames,labels,num_of_examples)

   return filenames,labels

def get_filenames_and_labels_gym_ub_s1(data_root,subset):

   DATA_PATH = data_root + 'subactions/'
   ANNO_PATH = data_root + 'annotations'

   filenames = []
   labels = []

   action_classes_to_include = list(range(74,89)) # set UB-S1  
   if 'train' in subset:

          for ln in open(f'{ANNO_PATH}/gym99_train.txt'):
                 file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                 if os.path.isfile(DATA_PATH+'/'+file_name):
                         if label in action_classes_to_include:
                              filenames.append(DATA_PATH+'/'+file_name)
                              label = label - 74 # off set labels to start from 0
                              labels.append(label)

   else:
          for ln in open(f'{ANNO_PATH}/gym99_val.txt'):
                 file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                 if os.path.isfile(DATA_PATH+'/'+file_name):
                         if label in action_classes_to_include:
                              filenames.append(DATA_PATH+'/'+file_name)
                              label = label - 74 # off set labels to start from 0
                              labels.append(label)

   return filenames,labels

def get_filenames_and_labels_gym_fx_s1(data_root,subset):

   DATA_PATH = data_root + 'subactions/'
   ANNO_PATH = data_root + 'annotations'

   filenames = []
   labels = []

   action_classes_to_include = [6,7,8,9,10,11,12,13,14,15,16]     # set FX-S1  
   if 'train' in subset:

          for ln in open(f'{ANNO_PATH}/gym99_train.txt'):
                 file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                 if os.path.isfile(DATA_PATH+'/'+file_name):
                         #print(file_name,label)
                         if label in action_classes_to_include:
                             filenames.append(DATA_PATH+'/'+file_name)
                             label = label - 6 
                             labels.append(label)

   else:
          for ln in open(f'{ANNO_PATH}/gym99_val.txt'):
                 file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                 if os.path.isfile(DATA_PATH+'/'+file_name):
                         if label in action_classes_to_include:
                             filenames.append(DATA_PATH+'/'+file_name)
                             label = label - 6 
                             labels.append(label)
   return filenames,labels

def get_filenames_and_labels_diving(data_root,subset):

   DATA_PATH = data_root + 'videos'
   ANNO_PATH = data_root + 'labels'

   filenames = []
   labels = []
   if 'train' in subset:
          video_list_path = f'{ANNO_PATH}/Diving48_V2_train.json' 
          #with open(video_list_path) as f:
          with open(video_list_path,encoding='utf-8') as f:
              video_infos = json.load(f)
              for video_info in video_infos:
                  video = video_info['vid_name']
                  video_name = f'{video}.avi'
                  label = int(video_info['label'])
                  if os.path.isfile(DATA_PATH+'/'+video_name):
                       filenames.append(DATA_PATH+'/'+video_name)
                       labels.append(label)

   else:
          video_list_path = f'{ANNO_PATH}/Diving48_V2_test.json' 
          #with open(video_list_path) as f:
          with open(video_list_path,encoding='utf-8') as f:
              video_infos = json.load(f)
              for video_info in video_infos:
                  video = video_info['vid_name']
                  video_name = f'{video}.avi'
                  label = int(video_info['label'])
                  if os.path.isfile(DATA_PATH+'/'+video_name):
                       filenames.append(DATA_PATH+'/'+video_name)
                       labels.append(label)

   return filenames,labels

def read_class_idx(annotation_dir: Path) -> Dict[str, str]:
    class_ind_path = annotation_dir+'/something-something-v2-labels.json'
    with open(class_ind_path) as f:
        class_dict = json.load(f)
    return class_dict

def get_filenames_and_labels_ssv2(data_root,subset):

        DATA_PATH = data_root + 'something-something-v2-videos_avi'
        ANNO_PATH = data_root + 'something-something-v2-annotations/'

        class_idx_dict = read_class_idx(ANNO_PATH)

        filenames = []
        labels = []
        if 'train' in subset:
               video_list_path = f'{ANNO_PATH}/something-something-v2-train.json' 
               #with open(video_list_path) as f:
               with open(video_list_path,encoding='utf-8') as f:
                   video_infos = json.load(f)
                   for video_info in video_infos:
                       video = int(video_info['id'])
                       video_name = f'{video}.avi'
                       class_name = video_info['template'].replace('[', '').replace(']', '')
                       class_index = int(class_idx_dict[class_name])
                       if os.path.isfile(DATA_PATH+'/'+video_name):
                            filenames.append(DATA_PATH+'/'+video_name)
                            labels.append(class_index)

        else:
               video_list_path = f'{ANNO_PATH}/something-something-v2-validation.json' 
               #with open(video_list_path) as f:
               with open(video_list_path,encoding='utf-8') as f:
                   video_infos = json.load(f)
                   for video_info in video_infos:
                       video = int(video_info['id'])
                       video_name = f'{video}.avi'
                       class_name = video_info['template'].replace('[', '').replace(']', '')
                       class_index = int(class_idx_dict[class_name])
                       if os.path.isfile(DATA_PATH+'/'+video_name):
                             filenames.append(DATA_PATH+'/'+video_name)
                             labels.append(class_index)
        return filenames,labels

