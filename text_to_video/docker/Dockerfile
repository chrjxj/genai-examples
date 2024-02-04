# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG INSTALL_ROOT=/workspace

ENV TZ=Asia/Shanghai DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y git tzdata \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/local/bin/pip /usr/bin/pip \
    && pip install protobuf==3.20.1 \
    && apt-get update -y && apt-get install -y ffmpeg \
    && pip install open_clip_torch pytorch_lightning scikit-video opencv-python==4.8.0.74 scikit-video gradio==3.14.0

WORKDIR ${INSTALL_ROOT}
