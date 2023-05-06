#!/bin/bash

CPU_PARENT=cnstark/pytorch:2.0.0-py3.9.12-ubuntu20.04
GPU_PARENT=cnstark/pytorch:2.0.0-py3.9.12-cuda11.8.0-ubuntu22.04

TAG=openrllab/openrl
VERSION=$(python -c "from openrl.__init__ import __version__;print(__version__)")

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
  TAG="${TAG}"
else
  PARENT=${CPU_PARENT}
  TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} -t ${TAG}:${VERSION} . -f docker/Dockerfile"
docker build --build-arg PARENT_IMAGE=${PARENT} -t ${TAG}:${VERSION} . -f docker/Dockerfile
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
  docker push ${TAG}:${VERSION}
  docker push ${TAG}:latest
fi
