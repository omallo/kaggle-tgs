#!/usr/bin/env bash

set -e

apt-get update >/dev/null
apt-get -y install python3-dev libsm-dev libxrender1 libxext6 zip git >/dev/null
rm -rf /var/lib/apt/lists/*

pip -q install virtualenv
virtualenv env --python=python3
. env/bin/activate

pip -q install -r requirements.txt

echo "commit: $(git rev-parse HEAD)" >/artifacts/out.log

python train.py | tee -a /artifacts/out.log

( cd /artifacts && zip -r logs.zip logs )
rm -rf /artifacts/logs
