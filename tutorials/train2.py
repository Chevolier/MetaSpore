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

# column_names = []
# with open(f'～/schema/column_name_mobivista.txt', 'r') as f:
#     for line in f:
#         column_names.append(line.split(' ')[1].strip())
# print(column_names)

column_names = ['_11001', '_11002', '_11003', '_11004', '_11007', '_11008', '_11021', '_11022', '_11023', '_11024', '_11041', '_11042', '_11043', '_11044', '_11045', '_11046', '_11061', '_11062', '_11063', '_11064', '_11065', '_11066', '_11081', '_11082', '_11083', '_11084', '_11085', '_11086', '_11601', '_11602', '_11603', '_12001', '_12002', '_12003', '_12004', '_12005', '_12006', '_20001', '_20002', '_20003', '_20101', '_20102', '_20201', '_20202', '_20203', '_20204', '_20205', '_20206', '_20207', '_20208', '_20209', '_20210', '_30001', '_30002', '_30003', '_30004', '_30005', '_30006', '_30201', '_30202', '_30203', '_30204', '_30205', '_30206', '_30207', '_40001', '_40002', '_40003', '_40004', '_40005', '_40201', '_40202', '_40203', '_40204', '_40205', '_40206', '_40207', '_40208', '_40209', '_40210', '_40211', '_40212', '_40213', '_40214', '_40215', '_40231', '_40301', '_40302', '_40303', '_40304', '_40305', '_40306', '_40307', '_40321', '_40322', '_40323', '_40324', '_50801', '_50802', '_50805', '_50806', '_50807', '_50810', '_51001', '_51002', '_51003', '_51004', '_51005', '_51006', '_51011', '_51012', '_51013', '_51014', '_51021', '_51022', '_51023', '_51024', '_51025', '_51026', '_51031', '_51032', '_51033', '_51034', '_51035', '_51036', '_51041', '_51042', '_51043', '_51044', '_51045', '_51046', '_52001', '_52002', '_52003', '_52004', '_52005', '_61101', '_61102', '_61103', '_70001', '_70002', '_70003', '_70004', '_70005', '_70006', '_70007', '_70008', '_70009', '_70010', '_70011', '_70031', '_70032', '_70033', '_70034', '_70035', '_70036', '_70037', '_70038', '_70039', '_70040', '_70041', '_70301', '_70302', '_70303', '_70304', '_70305', '_70306', '_70601', '_70602', '_70603', '_70604', '_70605', '_70606', '_70901', '_70902', '_70903', '_70904', '_70905', '_70906', '_72401', '_72402', '_80001', '_80002', '_80003', '_80004', '_80005', '_80006', '_80007', '_80008', '_80009', '_80010', '_80011', '_80012', '_80013', '_80014', '_80015', '_80016', '_80017', '_80018', '_80019', '_80020', '_80021', '_80022', '_80023', '_80024', '_80025', '_80026', '_80027', '_80028', '_80029', '_80030', '_80031', '_80032', '_80033', '_80034', '_80035', '_80036', '_80037', '_80038', '_80039', '_80040', '_80041', '_80042', '_80043', '_80044', '_80045', '_80046', '_80047', '_80048', '_80049', '_80050', '_80051', '_80052', '_80053', '_80054', '_80055', '_80056', '_80057', '_80058', '_80059', '_80060', '_80061', '_80062', '_80063', '_80064', '_80065', '_80066', '_80067', '_80068', '_80069', '_80070', '_80071', '_80072', '_80073', '_80074', '_80075', '_80076', '_80077', '_80078', '_80079', '_80080', '_80081', '_80082', '_80083', '_80084', '_80085', '_80086', '_80087', '_80088', '_80089', '_80090', '_80091', '_80092', '_80093', '_80094', '_80095', '_80096', '_80097', '_80098', '_80099', '_80100', '_80101', '_80102', '_80103', '_80104', '_80105', '_80106', '_80107', '_80108', '_80109', '_80110', '_80111', '_80112', '_80113', '_80114', '_80115', '_80125', '_80126', '_80127', '_80128', '_80129', '_80130', '_80131', '_80132', '_80133', '_80134', '_80135', '_80136', '_80137', '_80138', '_80139', '_80140', '_80141', '_80142', '_80143', '_80144', '_80145', '_80146', '_80147', '_80148', '_80149', '_80150', '_80151', 'label']

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