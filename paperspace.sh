apt-get update
apt-get -y install python3-dev libsm-dev libxrender1 libxext6
pip install virtualenv
virtualenv env --python=python3
. env/bin/activate
pip install -r requirements.txt
python train.py
rm -rf /var/lib/apt/lists/*
