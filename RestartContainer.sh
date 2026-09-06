#!/bin/bash

kubectl delete pods --selector app=ripdude \
&& sleep 3s \
&& kubectl logs --selector app=ripdude -f