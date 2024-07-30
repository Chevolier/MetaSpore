import metaspore as ms
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField, StringType, FloatType
from metaspore import _metaspore
from metaspore.url_utils import use_s3
from metaspore.file_utils import file_exists
from functools import reduce
from tqdm import tqdm
import time
import argparse
import os
import pandas as pd
import numpy as np

def str_to_bool(value):
    """Convert a string to a boolean."""
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}")

def split_value_weight(minibatch):
    start_time = time.time()
    def split(items):
        values = []
        weights = []
        for item in items:
            if '\003' in item:
                value, weight = item.split('\003')
            else:
                value, weight = '', '0'
            values.append(value)
            weights.append(float(weight))
        return values, weights

    values_dict = {}
    weights_dict = {}

    for column in minibatch.columns:
        if column == 'label':
            continue
        values_dict[f'{column}'] = []
        weights_dict[f'{column}_weight'] = []
        for items in minibatch[column]:
            try:
                # items is expected to be a numpy array or list
                if isinstance(items, (np.ndarray, list)):
                    values, weights = split(items)
                else:
                    values, weights = split([items])
                values_dict[f'{column}'].append(values)
                weights_dict[f'{column}_weight'].append(weights)
            except Exception as e:
                print(f"split_value_weight error, {e}, items: {items}")
                values_dict[f'{column}'].append([])
                weights_dict[f'{column}_weight'].append([])

    minibatch_value = pd.DataFrame(values_dict)
    minibatch_weight = pd.DataFrame(weights_dict)

    end_time = time.time()
    split_duration = end_time - start_time
    print(f"Type of minibatch: {type(minibatch)}, {minibatch.shape}, split duration: {split_duration:.3f} s.")

    return pd.concat([minibatch_value, minibatch_weight, minibatch[['label']]], axis=1)


def process_minibatch(iterator):
    for minibatch in iterator:
        yield split_value_weight(minibatch)

def generate_output_schema(columns):
    schema_fields = []
    for column in columns:
        if column == 'label':
            continue
        schema_fields.append(StructField(f'{column}', ArrayType(StringType()), True))
        schema_fields.append(StructField(f'{column}_weight', ArrayType(FloatType()), True))

    schema_fields.append(StructField('label', StringType(), True))
    return StructType(schema_fields)

def main(args):
    # Define Spark configuration and session
    spark_confs = {
            'spark.eventLog.enabled':'true',
            "spark.sql.files.ignoreCorruptFiles": "true",
            # 'spark.executor.memory': '30g',
            # 'spark.driver.memory': '15g',
            'spark.memory.fraction': args.spark_memory_fraction,
            # 'spark.sql.files.maxPartitionBytes': '256MB'
        }

    spark_session = ms.spark.get_session(local=args.local,
                                        batch_size=args.batch_size,
                                        worker_count=args.worker_count,
                                        server_count=args.server_count,
                                        worker_cpu=args.worker_cpu,
                                        server_cpu=args.server_cpu,
                                        worker_memory=args.worker_memory,
                                        server_memory=args.server_memory,
                                        coordinator_memory=args.coordinator_memory,
                                        log_level='WARN',
                                        spark_confs=spark_confs)

    with spark_session:

        column_name_path = use_s3('s3://mv-mtg-di-for-poc-datalab/schema/column_name_mobivista.txt')
        if not file_exists(column_name_path):
            raise RuntimeError(f"combine schema file {column_name_path!r} not found")
        columns = _metaspore.stream_read_all(column_name_path)
        columns = [column.split(' ')[1].strip() for column in columns.decode('utf-8').split('\n') if column.strip()]
        print(f"column_names: {columns}")

        # file_names = [f'part-{str(i).zfill(5)}-1e73cc51-9b17-4439-9d71-7d505df2cae3-c000.snappy.orc' for i in range(args.num_files)]
        # train_dataset_path = [args.file_base_path + file_name for file_name in file_names]

        train_dataset_path = [args.file_base_path + f"/{i:02d}/" for i in range(args.num_files)]

        # Load the dataset
        train_dataset = ms.input.read_s3_csv(spark_session, 
                                            train_dataset_path, 
                                            format='orc',
                                            shuffle=args.shuffle,
                                            delimiter='\t', 
                                            multivalue_delimiter="\001", 
                                            column_names=columns,
                                            multivalue_column_names=columns[:-1])

        num_partitions = train_dataset.rdd.getNumPartitions()
        print(f"Number of orcs: {args.num_files}, total number of rows: {train_dataset.count()}")
        print(f"Number of partitions: {num_partitions}")
        
        # train_dataset.printSchema()
        # train_dataset.show(5)

        start_time = time.time()

        output_schema = generate_output_schema(train_dataset.columns)
        processed_df = train_dataset.mapInPandas(process_minibatch, schema=output_schema)

        # Save the processed DataFrame as ORC files
        # num_partitions = args.num_files  # Set to 1 to ensure a single file or adjust based on your data size
        processed_df_repartitioned = processed_df.repartition(num_partitions)

        # processed_df_repartitioned.printSchema()
        # processed_df_repartitioned.show(5)

        print(f"Total number of processed rows: {processed_df_repartitioned.count()}")
        print(f"Total number of processed partitions: {processed_df_repartitioned.rdd.getNumPartitions()}")
      
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        processed_df_repartitioned.write.mode('overwrite').format(args.output_format).save(args.output_dir)

        time_cost = time.time() - start_time

        print(f"Processing time cost: {time_cost:.2f} s.")

        # Stop the Spark session
        spark_session.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--column-name-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/schema/column_name_mobivista.txt')
    parser.add_argument('--combine-schema-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/schema/combine_schema_mobivista.txt')
    parser.add_argument('--file-base-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/2024/06/14/00/')
    parser.add_argument('--test-dataset-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/2024/06/15/00/part-00000-f79b9ee6-aaf5-4117-88d5-44eea69dcea3-c000.snappy.orc')    
    parser.add_argument('--output-dir', type=str, default='/home/ubuntu/data/processed/orcs/')    
    parser.add_argument('--output-format', type=str, default='orc')    
    parser.add_argument('--num-files', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--server-count', type=int, default=1)
    parser.add_argument('--worker-cpu', type=int, default=1)
    parser.add_argument('--server-cpu', type=int, default=1)
    parser.add_argument('--worker-memory', type=str, default='15G')
    parser.add_argument('--server-memory', type=str, default='15G')    
    parser.add_argument('--coordinator-memory', type=str, default='15G') 
    parser.add_argument('--spark-memory-fraction', type=str, default='0.6')       
    parser.add_argument('--experiment-name', type=str, default='0.1')
    parser.add_argument('--training-epochs', type=int, default=1)
    parser.add_argument('--shuffle', type=str_to_bool, default=False,
                    help="Whether to shuffle the dataset. Use 'true' or 'false'.")
    parser.add_argument('--shuffle-training-dataset', type=str_to_bool, default=False,
                    help="Whether to shuffle the dataset. Use 'true' or 'false'.")
    parser.add_argument('--local', action='store_true')  # Use store_true for the local parameter

    args = parser.parse_args()

    main(args)