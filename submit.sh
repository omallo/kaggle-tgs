#!/usr/bin/env bash

apt-get -qq update
apt-get -qq -y install python3-dev libsm-dev libxrender1 libxext6 zip
rm -rf /var/lib/apt/lists/*

pip install virtualenv
virtualenv env --python=python3
. env/bin/activate

pip install -r requirements.txt

python submit.py | tee -a /artifacts/out.log
