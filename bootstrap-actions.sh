#!/bin/bash

echo "Installing metaspore related packages ..."

aws s3 cp s3://spark-emr-data/wheels/metaspore-1.2.0-cp39-cp39-linux_x86_64.whl .
sudo pip install metaspore-1.2.0-cp39-cp39-linux_x86_64.whl
sudo pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
sudo pip install https://ks3-cn-beijing.ksyuncs.com/dmetasoul-bucket/releases/spark/pyspark-3.1.2.265f9ad4ee-py2.py3-none-any.whl
sudo pip install numpy==1.23.5

echo "Updating java to 11 ..."

sudo update-alternatives --set java /usr/lib/jvm/java-11-amazon-corretto.x86_64/bin/java
sudo update-alternatives --set javac /usr/lib/jvm/java-11-amazon-corretto.x86_64/bin/javac

# sudo mkdir -p /etc/alternatives/jre/bin
# sudo ln -s /usr/lib/jvm/java-11-amazon-corretto.x86_64/bin/java /etc/alternatives/jre/bin/java

echo 'export JAVA_HOME=/usr/lib/jvm/java-11-amazon-corretto.x86_64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc


echo "Installing awscli ..."

sudo pip install awscli --ignore-installed six
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

sudo yum update -y
sudo yum install git -y

echo "Configure libstdc++.so.6"

sudo aws s3 cp s3://spark-emr-data/wheels/libstdc++.so.6.0.30 /lib64/
sudo rm /lib64/libstdc++.so.6
sudo ln -s /lib64/libstdc++.so.6.0.30 /lib64/libstdc++.so.6

mkdir -p /tmp/spark-events

# echo 'export SPARK_HOME=/usr/lib/spark' >> ~/.bashrc
# source ~/.bashrc

# echo 'export HADOOP_CONF_DIR=/etc/hadoop/conf' >> ~/.bashrc
# source ~/.bashrc

# aws s3 cp s3://spark-emr-data/jars/spark-3.1.2-bin-hadoop3.2.tgz ~/
# tar zxvf ~/spark-3.1.2-bin-hadoop3.2.tgz -C ~/
# sudo cp ~/spark-3.1.2-bin-hadoop3.2/jars/spark-yarn* /usr/local/lib/python3.9/site-packages/pyspark/jars/
# sudo cp ~/spark-3.1.2-bin-hadoop3.2/jars/hadoop-* /usr/local/lib/python3.9/site-packages/pyspark/jars/

sudo aws s3 cp s3://spark-emr-data/jars/spark-yarn_2.12-3.1.2.jar /usr/local/lib/python3.9/site-packages/pyspark/jars/
sudo aws s3 cp s3://spark-emr-data/jars/hadoop-yarn-server-web-proxy-2.10.1.jar /usr/local/lib/python3.9/site-packages/pyspark/jars/

echo "Bootstrap action completed successfully"
