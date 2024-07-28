import metaspore as ms
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField, StringType
from metaspore import _metaspore
from metaspore.url_utils import use_s3
from metaspore.file_utils import file_exists
from functools import reduce
from tqdm import tqdm
import time
import argparse

def str_to_bool(value):
    """Convert a string to a boolean."""
    if value.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif value.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got {value}")

def main(args):
    # Define Spark configuration and session
    spark_confs = {
            'spark.eventLog.enabled':'true',
            "spark.sql.files.ignoreCorruptFiles": "true",
            # 'spark.executor.memory': '30g',
            # 'spark.driver.memory': '15g',
            'spark.memory.fraction': args.spark_memory_fraction
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

        file_names = [f'part-{str(i).zfill(5)}-1e73cc51-9b17-4439-9d71-7d505df2cae3-c000.snappy.orc' for i in range(args.num_files)]
        train_dataset_path = [args.file_base_path + file_name for file_name in file_names]

        # train_dataset_path = [args.file_base_path + f"/{i:02d}/" for i in range(args.num_files)]

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

        # Define UDF to process rows
        def process_row(column_values):
            feature_hashes = set()
            for value in column_values:
                if '\003' in value:
                    hash_val, weight = value.split('\003')
                else:
                    hash_val, weight = '', 0
                feature_hashes.add(hash_val)

            feature_values = len(column_values)
            return (feature_values, list(feature_hashes))

        # Define schema for UDF output
        schema = StructType([
            StructField("feature_values", IntegerType(), True),
            StructField("hashes", ArrayType(StringType()), True)
        ])

        # Register the UDF
        process_row_udf = udf(process_row, schema)
        start_time = time.time()

        # Apply UDF on each column and process the dataset
        for column in columns[:-1]:
            train_dataset = train_dataset.withColumn(f'{column}_processed', process_row_udf(col(column)))

        # Extract the processed columns
        result_columns = [f'{column}_processed' for column in columns[:-1]]
        processed_df = train_dataset.select(*result_columns)
        processed_df.show()

        end_time_process = time.time()
        time_cost_process = end_time_process - start_time

        # Obtain the total number of different hash values for each column and across all columns
        print(f"Starting counting hashes")

        # Collect distinct hashes for each column separately
        distinct_hash_counts = {}
        all_hashes_rdd = spark_session.sparkContext.emptyRDD()

        for col_name in tqdm(result_columns, total=len(result_columns)):
            col_hashes_rdd = processed_df.select(explode(col(f"{col_name}.hashes")).alias("hashes")).rdd.map(lambda row: row["hashes"]).distinct()
            distinct_hash_counts[col_name] = col_hashes_rdd.count()
            all_hashes_rdd = all_hashes_rdd.union(col_hashes_rdd)

        end_time_union = time.time()
        time_cost_union = end_time_union - end_time_process

        print("Finished union RDDs!")

        # Get the total distinct hashes across all columns
        total_distinct_hashes = all_hashes_rdd.distinct().count()

        end_time_count = time.time()
        time_cost_count = end_time_count - end_time_union

        print(f"Processing time cost: {time_cost_process:.2f} s, Union time cost: {time_cost_union:.2f} s, Count time cost: {time_cost_count:.2f} s.")
        print(f"Distinct hash counts per column: {distinct_hash_counts}")
        print(f"Total distinct hashes across all columns: {total_distinct_hashes}")

        # Stop the Spark session
        spark_session.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--column-name-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/schema/column_name_mobivista.txt')
    parser.add_argument('--combine-schema-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/schema/combine_schema_mobivista.txt')
    parser.add_argument('--file-base-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/2024/06/14/00/')
    parser.add_argument('--test-dataset-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/2024/06/15/00/part-00000-f79b9ee6-aaf5-4117-88d5-44eea69dcea3-c000.snappy.orc')    
    parser.add_argument('--model-out-path', type=str, default='s3://mv-mtg-di-for-poc-datalab/output/dev/model_out/')    
    parser.add_argument('--num-files', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--server-count', type=int, default=1)
    parser.add_argument('--worker-cpu', type=int, default=1)
    parser.add_argument('--server-cpu', type=int, default=1)
    parser.add_argument('--worker-memory', type=str, default='5G')
    parser.add_argument('--server-memory', type=str, default='5G')    
    parser.add_argument('--coordinator-memory', type=str, default='5G') 
    parser.add_argument('--spark-memory-fraction', type=str, default='0.6')       
    parser.add_argument('--experiment-name', type=str, default='0.1')
    parser.add_argument('--training-epochs', type=int, default=1)
    parser.add_argument('--shuffle', type=str_to_bool, default=False,
                    help="Whether to shuffle the dataset. Use 'true' or 'false'.")
    parser.add_argument('--shuffle-training-dataset', type=str_to_bool, default=False,
                    help="Whether to shuffle the dataset. Use 'true' or 'false'.")
    parser.add_argument('--local', action='store_true')  # Use store_true for the local parameter

    args = parser.parse_args()