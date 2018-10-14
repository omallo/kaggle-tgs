#!/usr/bin/env bash

set -e
set -o pipefail

trap archive_artifacts EXIT

function install_dependencies() {
  apt-get update >/dev/null
  apt-get -y install python3-dev libsm-dev libxrender1 libxext6 zip git >/dev/null
  rm -rf /var/lib/apt/lists/*

  pip -q install virtualenv
  virtualenv env --python=python3
  . env/bin/activate

  pip -q install -r requirements.txt
}

function archive_artifacts() {
  if [ -d /artifacts/logs ]
  then
    ( cd /artifacts && zip -q -r logs.zip logs )
    rm -rf /artifacts/logs
  fi

  rm -rf /storage/models/tgs/${RUN_NAME}
  mkdir -p /storage/models/tgs/${RUN_NAME}
  cp -r /artifacts/* /storage/models/tgs/${RUN_NAME}
}

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

install_dependencies

printf "commit: $(git rev-parse HEAD)\n\n" | tee -a /artifacts/out.log

python -m cProfile -o /artifacts/train.prof train.py $* 2>/artifacts/err.log | tee -a /artifacts/out.log
