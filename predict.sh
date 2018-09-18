#!/usr/bin/env bash

apt-get update
apt-get -y install python3-dev libsm-dev libxrender1 libxext6 zip
rm -rf /var/lib/apt/lists/*

pip install virtualenv
virtualenv env --python=python3
. env/bin/activate

pip install -r requirements.txt

python predict.py | tee -a /artifacts/out.log
