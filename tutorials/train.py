import argparse
import torch
import torch.nn as nn
import metaspore as ms
import pyspark
from metaspore import _metaspore
from metaspore.url_utils import is_url
from metaspore.url_utils import use_s3
from metaspore.file_utils import file_exists

def nansum(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x).sum()


def log_loss(yhat, y):
    return nansum(-(y * (yhat + 1e-12).log() + (1 - y) *
                    (1 - yhat + 1e-12).log()))

# 自定义的主函数入口
class DNNModelMain(nn.Module):
    def __init__(self, schema_path): # feature_config_file
        super().__init__()
        self._embedding_size = 16
        # self._schema_dir = schema_dir # root_dir + '/schema/'
        # self._column_name_path = self._schema_dir + 'column_name_mobivista.txt'
        self._combine_schema_path = schema_path # self._schema_dir + 'combine_schema_mobivista.txt'
        # self.feature_config_file = feature_config_file  # TODO not used
        self._sparse = ms.EmbeddingSumConcat(
            self._embedding_size,
            # combine_schema_source=self._column_name_path,
            combine_schema_file_path=self._combine_schema_path,
            # enable_feature_gen=True,
            # feature_config_file=feature_config_file,
            # enable_fgs=False
        )
        self._sparse.updater = ms.FTRLTensorUpdater(alpha=0.01)
        self._sparse.initializer = ms.NormalTensorInitializer(var=0.001)
        extra_attributes = {
            "enable_fresh_random_keep": True,
            "fresh_dist_range_from": 0, 
            "fresh_dist_range_to": 1000,
            "fresh_dist_range_mean": 950,
            "enable_feature_gen": True,
            "use_hash_code": False
        }
        self._sparse.extra_attributes = extra_attributes
        feature_count = self._sparse.feature_count
        feature_dim = self._sparse.feature_count * self._embedding_size

        self._gateEmbedding = GateEmbedding(feature_dim, feature_count, self._embedding_size)
        self._h1 = nn.Linear(feature_dim, 1024)
        self._h2 = FourChannelHidden(1024, 512)
        self._h3 = FourChannelHidden(512, 256)
        self._h4 = nn.Linear(256, 1)

        self._bn = ms.nn.Normalization(feature_dim, momentum=0.01, eps=1e-5, affine=True)
        self._zero = torch.zeros(1, 1)
        self.act0 = nn.Sigmoid()

    def forward(self, x):
        emb = self._sparse(x)
        bno = self._bn(emb)
        d = self._gateEmbedding(bno)
        o = self._h1(d)
        r, s1, s2, s3 = self._h2(o, self._zero, self._zero, self._zero)
        r, s1, s2, s3 = self._h3(r, s1, s2, s3)
        return self.act0(self._h4(r))


class FourChannelHidden(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.wc2 = nn.Linear(int(in_size / 4), int(in_size / 4))
        self.wc3 = nn.Linear(int(in_size), int(in_size - in_size / 4 * 3))
        self.w = nn.Linear(int(in_size + int(in_size / 4) * 2) + int(in_size - int(in_size / 4) * 3) + 3, out_size)
        self.act1 = nn.Tanh()
        self.act = nn.ReLU()
        self.fl = int(in_size / 4)

    def forward(self, input, i1, i2, i3):
        f0 = input[:, :self.fl]
        f1 = input[:, self.fl:self.fl * 2]
        f2 = input[:, self.fl * 2:self.fl * 3]
        f3 = input[:, self.fl * 3:]

        c1 = self.act1(f0 * f1) * f1
        c2 = self.act1(self.wc2(f2) * f2)
        c3 = self.act1(f3 * self.wc3(input))

        s1 = torch.sum(c1, 1, True) + i1
        s2 = torch.sum(c2, 1, True) + i2
        s3 = torch.sum(c3, 1, True) + i3

        return self.act(self.w(torch.cat((input, c1, c2, c3, s1, s2, s3), 1))), s1, s2, s3


class GateEmbedding(nn.Module):
    def __init__(self, in_size, out_size, emb_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_size, out_size)
        self.out_size = out_size
        self.emb_size = emb_size
        self.act2 = nn.Sigmoid()

    def forward(self, input):
        gate = self.act2(self.layer1(input))
        gate_reshape = torch.reshape(gate, (-1, self.out_size, 1))
        input_reshape = torch.reshape(input, (-1, self.out_size, self.emb_size))
        return (gate_reshape * input_reshape).reshape(-1, self.out_size * self.emb_size)
    

def train(args):
    spark_confs = {
        'spark.eventLog.enabled':'true',
        "spark.sql.files.ignoreCorruptFiles": "true"
        # 'spark.executor.memory': '30g',
        # 'spark.driver.memory': '15g',
    }

    spark_session = ms.spark.get_session(local=args.local,
                                        batch_size=args.batch_size,
                                        worker_count=args.worker_count,
                                        server_count=args.server_count,
                                        worker_cpu=args.worker_cpu,
                                        server_cpu=args.server_cpu,
                                        log_level='WARN',
                                        spark_confs=spark_confs)

    with spark_session:
        module = DNNModelMain(args.combine_schema_path)

        estimator = ms.PyTorchEstimator(module=module,
                                        worker_count=args.worker_count,
                                        server_count=args.server_count,
                                        model_out_path=args.model_out_path,
                                        experiment_name='0.1',
                                        input_label_column_name='label',
                                        training_epoches=args.training_epochs,
                                        shuffle_training_dataset=False)

        column_name_path = use_s3(args.column_name_path)
        if not file_exists(column_name_path):
            raise RuntimeError(f"combine schema file {column_name_path!r} not found")
        column_names = _metaspore.stream_read_all(column_name_path)
        column_names = [column.split(' ')[1].strip() for column in column_names.decode('utf-8').split('\n') if column.strip()]
        print(f"column_names: {column_names}")
        
        file_names = [f'part-{str(i).zfill(5)}-1e73cc51-9b17-4439-9d71-7d505df2cae3-c000.snappy.orc' for i in range(args.num_files)]
        train_dataset_path = [args.file_base_path + file_name for file_name in file_names]

        train_dataset = ms.input.read_s3_csv(spark_session, 
                                            train_dataset_path, 
                                            format='orc',
                                            shuffle=False,
                                            delimiter='\t', 
                                            multivalue_delimiter="\001", 
                                            column_names=column_names,
                                            multivalue_column_names=column_names[:-1])

        # print(f"Number of training samples: {train_dataset.count()}")
        # print("Start training ...")

        model = estimator.fit(train_dataset)

        print("Finished training!")
        print("Start testing ...")

        test_dataset = ms.input.read_s3_csv(spark_session, 
                                            args.test_dataset_path, 
                                            format='orc',
                                            delimiter='\t', 
                                            multivalue_delimiter="\001", 
                                            column_names=column_names,
                                            multivalue_column_names=column_names[:-1])
        
        result = model.transform(test_dataset)
        result_pd = result.toPandas()
        print(f"Negative sample results: {result_pd[result_pd['label']==0].head()}")
        print(f"Positive sample results: {result_pd[result_pd['label']==1].head()}")
        
        evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator()
        test_auc = evaluator.evaluate(result)
        print('test_auc: %g' % test_auc)
        print("Finished testing!")


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
    parser.add_argument('--experiment-name', type=str, default='0.1')
    parser.add_argument('--training-epochs', type=int, default=1)
    parser.add_argument('--local', action='store_true')  # Use store_true for the local parameter

    args = parser.parse_args()

    train(args)
