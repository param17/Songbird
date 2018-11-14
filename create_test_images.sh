#!/usr/bin/env bash

path_to_raw_imgs=$1
num_img_per_cat=$2

mkdir test_imgs

for category in $(seq 0 1 5)
do
        image_cat_path=${path_to_raw_imgs}'/'${category}

        for file in $(ls ${image_cat_path} | shuf -n ${num_img_per_cat})
        do
            cp ${image_cat_path}'/'${file} test_imgs/
            echo 'Using image: '${file}
        done
done