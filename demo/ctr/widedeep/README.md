
# Run the demo/ctr/widedeep

```bash

export MY_S3_BUCKET='s3a://sagemaker-us-west-2-452145973879/datasets/'
envsubst < fg.yaml > fg.yaml.dev 

envsubst < match_dataset_cf.yaml > match_dataset.yaml.dev 
nohup python match_dataset_cf.py --conf match_dataset.yaml.dev --verbose > log_match.out 2>&1 &


envsubst < match_dataset_negsample_10.yaml > match_dataset_negsample_10.yaml.dev 
nohup python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml.dev --verbose > log_match_neg.out 2>&1 &

envsubst < rank_dataset.yaml > rank.yaml.dev

nohup python rank_dataset.py --conf rank.yaml.dev --verbose > log_rank.out 2>&1 &


cd ctr/widedeep/conf
envsubst < widedeep_ml_1m.yaml > widedeep_ml_1m.yaml.dev


cd MetaSpore/demo/ctr/widedeep
python widedeep.py --conf conf/widedeep_ml_1m.yaml.dev

```