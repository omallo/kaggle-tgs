#!/usr/bin/env bash

set -e
set -o pipefail

apt-get update >/dev/null
apt-get -y install python3-dev libsm-dev libxrender1 libxext6 zip git >/dev/null
rm -rf /var/lib/apt/lists/*

pip -q install virtualenv
virtualenv env --python=python3
. env/bin/activate

pip -q install -r requirements.txt

printf "commit: $(git rev-parse HEAD)\n\n" | tee -a /artifacts/out.log

RUN_NAME=$1

while (( "$#" ))
do
  case "$1" in
  --)
    shift
    break
    ;;
  *)
    shift
    ;;
  esac
done

trap 'on_exit' EXIT

python -m cProfile -o /artifacts/train.prof train.py $* 2>/artifacts/err.log | tee -a /artifacts/out.log

function on_exit() {
  if [ -f /artifacts/logs ]
  then
    ( cd /artifacts && zip -r logs.zip logs )
    rm -rf /artifacts/logs
  fi

  rm -rf /storage/models/tgs/${RUN_NAME}
  mkdir -p /storage/models/tgs/${RUN_NAME}
  cp -r /artifacts/* /storage/models/tgs/${RUN_NAME}
}
