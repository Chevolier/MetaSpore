import argparse
import os
import torch
import torch.nn as nn
import metaspore as ms
from pyspark.sql import SparkSession

# Define the model classes here...

def nansum(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x).sum()

def log_loss(yhat, y):
    return nansum(-(y * (yhat + 1e-12).log() + (1 - y) * (1 - yhat + 1e-12).log()))

def train(args):
    ROOT_DIR = args.root_dir
    
    # Initialize Spark session
    spark_conf = {
        'spark.eventLog.enabled': 'true',
        'spark.executor.memory': '20g',
        'spark.driver.memory': '10g',
    }
    
    spark = SparkSession.builder \
        .appName("SageMaker Training") \
        .config(conf=spark_conf) \
        .getOrCreate()
    
    # Define the model here...
    module = DNNModelMain()
    
    estimator = ms.PyTorchEstimator(
        module=module,
        worker_count=args.worker_count,
        server_count=args.server_count,
        model_out_path=args.model_out_path,
        experiment_name=args.experiment_name,
        input_label_column_name='label',
        training_epoches=args.training_epochs,
        shuffle_training_dataset=True
    )
    
    # Read column names
    column_names = []
    with open(args.column_name_path, 'r') as f:
        for line in f:
            column_names.append(line.split(' ')[1].strip())
    
    # Prepare dataset paths
    file_base_path = args.file_base_path
    num_files = args.num_files
    file_names = [f'part-{str(i).zfill(5)}-1e73cc51-9b17-4439-9d71-7d505df2cae3-c000.snappy.orc' for i in range(num_files)]
    train_dataset_path = [file_base_path + file_name for file_name in file_names]
    
    train_dataset = ms.input.read_s3_csv(
        spark,
        train_dataset_path,
        format='orc',
        shuffle=False,
        delimiter='\t',
        multivalue_delimiter="\001",
        column_names=column_names,
        multivalue_column_names=column_names[:-1]
    )
    
    model = estimator.fit(train_dataset)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='s3://mv-mtg-di-for-poc-datalab')
    parser.add_argument('--column-name-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/schema/column_name_mobivista.txt')
    parser.add_argument('--combine-schema-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/schema/combine_schema_mobivista.txt')
    parser.add_argument('--file-base-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/2024/06/14/00/')
    parser.add_argument('--num-files', type=int, default=100)
    parser.add_argument('--worker-count', type=int, default=100)
    parser.add_argument('--server-count', type=int, default=200)
    parser.add_argument('--model-out-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/output/dev/model_out/')
    parser.add_argument('--experiment-name', type=str, default='0.1')
    parser.add_argument('--training-epochs', type=int, default=1)
    
    args = parser.parse_args()
    train(args)