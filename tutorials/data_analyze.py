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

# Define Spark configuration and session
spark_confs = {
    'spark.eventLog.enabled': 'true',
    'spark.executor.memory': '10g',
    'spark.driver.memory': '10g',
    "spark.driver.maxResultSize": "10g",
    "spark.sql.files.ignoreCorruptFiles": "true"
}

spark_session = ms.spark.get_session(
    local=False,  # Remove local=True to run in cluster mode
    batch_size=500,
    worker_count=8,
    server_count=4,
    worker_cpu=5,
    server_cpu=5,
    log_level='WARN',
    spark_confs=spark_confs
)

column_name_path = use_s3('s3://mv-mtg-di-for-poc-datalab/schema/column_name_mobivista.txt')
if not file_exists(column_name_path):
    raise RuntimeError(f"combine schema file {column_name_path!r} not found")
columns = _metaspore.stream_read_all(column_name_path)
columns = [column.split(' ')[1].strip() for column in columns.decode('utf-8').split('\n') if column.strip()]
print(f"column_names: {columns}")

file_base_path = 's3://mv-mtg-di-for-poc-datalab/2024/06/14/00/'
num_files = 10
file_names = [f'part-{str(i).zfill(5)}-1e73cc51-9b17-4439-9d71-7d505df2cae3-c000.snappy.orc' for i in range(num_files)]
train_dataset_path = [file_base_path + file_name for file_name in file_names]

train_dataset_path = 's3://mv-mtg-di-for-poc-datalab/2024/06/14/00/'

# Load the dataset
train_dataset = ms.input.read_s3_csv(
    spark_session, 
    train_dataset_path, 
    format='orc',
    shuffle=False,
    delimiter='\t', 
    multivalue_delimiter="\001", 
    column_names=columns,
    multivalue_column_names=columns[:-1]
)

print(f"Number of orcs: {num_files}, total number of rows: {train_dataset.count()}")

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