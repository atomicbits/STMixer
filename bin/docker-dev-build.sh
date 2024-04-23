#!/bin/bash

# Run this script from within the docker folder containing the Dockerfile 

image_name="stmixer:v1"

#docker rmi -f "$image_name"

docker build -t "$image_name" -f docker/Dockerfile .