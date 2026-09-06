#!/bin/bash

dir="${0%/*}"
tag=${1:-latest}

sudo nerdctl --namespace k8s.io build -t kabanowster/ripdude:$tag -f $dir/Dockerfile $dir