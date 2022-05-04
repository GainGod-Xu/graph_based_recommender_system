#!/bin/bash
cd ./models/CompGCN || exit
sh preprocess.sh
cp -r data/* ../../data/