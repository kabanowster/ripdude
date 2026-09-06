#!/bin/bash

dir="${0%/*}"

sudo nerdctl --namespace k8s.io build -t cognee:falkor -f $dir/Dockerfile $dir