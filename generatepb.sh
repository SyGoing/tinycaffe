#!/usr/bin/env bash

protoc -I="./src" --cpp_out="./src" --python_out="." "./src/caffe.proto"
