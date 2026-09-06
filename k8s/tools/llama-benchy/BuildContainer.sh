#!/bin/bash

dir="${0%/*}"

sudo nerdctl --namespace k8s.io build -t llama-benchy:latest -f $dir/Dockerfile .