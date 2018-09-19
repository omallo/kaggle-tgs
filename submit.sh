#!/usr/bin/env bash

apt-get update >/dev/null
apt-get -y install python3-dev libsm-dev libxrender1 libxext6 zip >/dev/null
rm -rf /var/lib/apt/lists/*

pip -q --no-color install virtualenv
virtualenv env --python=python3
. env/bin/activate

pip -q install -r requirements.txt

python submit.py | tee -a /artifacts/out.log
