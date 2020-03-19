from __future__ import print_function

import os
# print(f'\n\n\nCURRENT DIRECTORY: {os.getcwd()}\n\n\n')
# print(f'current file: {__file__}')
# print(f'listdir cwd: {os.listdir()}')
# print(f"listdir /.sagemaker/mms/models/model: {os.listdir('/.sagemaker/mms/models/model')}")
# # traverse root directory, and list directories as dirs and files as files
# print('walking /.sagemaker/mms/models/model...')
# for root, dirs, files in os.walk('/.sagemaker/mms/models/model'):
#     path = root.split(os.sep)
#     print((len(path) - 1) * '---', os.path.basename(root))
#     for file in files:
#         print(len(path) * '---', file)
# print('walking current directory...')
# for root, dirs, files in os.walk(os.getcwd()):
#     path = root.split(os.sep)
#     print((len(path) - 1) * '---', os.path.basename(root))
#     for file in files:
#         print(len(path) * '---', file)
#         print(len(path) * '---', file)
# print('walking /opt/ml/model/code/...')
# for root, dirs, files in os.walk('/opt/ml/model/code/'):
#     path = root.split(os.sep)
#     print((len(path) - 1) * '---', os.path.basename(root))
#     for file in files:
#         print(len(path) * '---', file)        
# print(os.environ)

import argparse
import logging
import os
import json

import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '/opt/ml/code/package'))
from autogluon import TabularPrediction as task
import pandas as pd
import pickle
from io import StringIO

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    net = task.load(model_dir)
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """

    f = open("/opt/ml/model/code/list.txt","r")
    list_row = []
    for x in f:
        list_row.append(x.rstrip("\n"))
    f.close()
    #logging.info(print(data))
    data_buf = StringIO(data)
    
    df_parsed = pd.read_csv(data_buf, header=None, names=list_row)
    prediction = net.predict(df_parsed)

    response_body = json.dumps(prediction.tolist())

    return response_body, output_content_type


# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--filename', type=str, default='ESOL_train.csv')

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    
    parser.add_argument('--target', type=str, default='label')

    return parser.parse_args()


if __name__ == "__main__":  
    args = parse_args()