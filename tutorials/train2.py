ROOT_DIR = 's3://mv-mtg-di-for-poc-datalab'

import metaspore as ms
import torch
import torch.nn as nn


def nansum(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x).sum()


def log_loss(yhat, y):
    return nansum(-(y * (yhat + 1e-12).log() + (1 - y) *
                    (1 - yhat + 1e-12).log()))

# 自定义的主函数入口
class DNNModelMain(nn.Module):
    def __init__(self, ): # feature_config_file
        super().__init__()
        self._embedding_size = 16
        self._schema_dir = ROOT_DIR + '/schema/'
        self._column_name_path = self._schema_dir + 'column_name_mobivista.txt'
        self._combine_schema_path = self._schema_dir + 'combine_schema_mobivista.txt'
        # self.feature_config_file = feature_config_file  # TODO not used
        self._sparse = ms.EmbeddingSumConcat(
            self._embedding_size,
            combine_schema_source=self._column_name_path,
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
        
        # print(f"self._sparse._data.type: {type(self._sparse._data)}, self._sparse._data.shape: {self._sparse._data.shape}") 
        # print(f"x.type: {type(x)}, x.shape: {x.shape}, x: {x}")
        # print(f"emb.type: {type(emb)}, emb.shape: {emb.shape}, ") # emb: {emb}
        # print(f"bno.type: {type(bno)}, bno.shape: {bno.shape}, ")  # bno: {bno}
        
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
    
# module = DemoModule()
# module = DNNModelMain('schema/combine_schema_mobivista.txt')
module = DNNModelMain()

model_out_path = ROOT_DIR + '/output/dev/model_out/'
estimator = ms.PyTorchEstimator(module=module,
                                worker_count=8,
                                server_count=8,
                                model_out_path=model_out_path,
                                experiment_name='0.1',
                                input_label_column_name='label',
                                training_epoches=1,
                                shuffle_training_dataset=True)

column_names = []
with open(f'./schema/column_name_mobivista.txt', 'r') as f:
    for line in f:
        column_names.append(line.split(' ')[1].strip())
print(column_names)

# train_dataset_path = ROOT_DIR + '/data/train/day_0_0.001_train.csv'

file_base_path = 's3://mv-mtg-di-for-poc-datalab/2024/06/14/00/'
num_files = 1
file_names = [f'part-{str(i).zfill(5)}-1e73cc51-9b17-4439-9d71-7d505df2cae3-c000.snappy.orc' for i in range(num_files)]
train_dataset_path = [file_base_path + file_name for file_name in file_names]

# train_dataset_path = 's3://mv-mtg-di-for-poc-datalab/2024/06/14/00/part-00000-1e73cc51-9b17-4439-9d71-7d505df2cae3-c000.snappy.orc'
# train_dataset_path = 's3://mv-mtg-di-for-poc-datalab/2024/06/14/00/'

spark_confs = {
    'spark.eventLog.enabled':'true',
    'spark.executor.memory': '20g',
    'spark.driver.memory': '10g',
}

spark_session = ms.spark.get_session(local=True,
                                     batch_size=100,
                                     worker_count=estimator.worker_count,
                                     server_count=estimator.server_count,
                                     log_level='WARN',
                                     spark_confs=spark_confs)

train_dataset = ms.input.read_s3_csv(spark_session, 
                                     train_dataset_path, 
                                     format='orc',
                                     shuffle=False,
                                     delimiter='\t', 
                                     multivalue_delimiter="\001", 
                                     column_names=column_names,
                                     multivalue_column_names=column_names[:-1])

# train_dataset = spark_session.read.orc(train_dataset_path)

# train_dataset.printSchema()
train_dataset.count()

model = estimator.fit(train_dataset)