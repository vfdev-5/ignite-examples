#!/usr/bin/python3.5
# -*- coding:utf-8 -*-
# Created Time: Fri 02 Mar 2018 03:58:07 PM CST
# Purpose: download image
# Mail: tracyliang18@gmail.com
# Adapted to python 3 by Aloisio Dourado in Sun Mar 11 2018

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing
import requests
from PIL import Image
from io import BytesIO
from tqdm  import tqdm
import json


def ParseData(data_file):
    ann = {}
    if 'train' in data_file or 'validation' in data_file:
        _ann = json.load(open(data_file))['annotations']
        for a in _ann:
            ann[a['image_id']] = a['label_id']

    key_url_list = []
    j = json.load(open(data_file))
    images = j['images']
    for item in images:
        assert len(item['url']) == 1
        url = item['url'][0]
        id_ = item['image_id']
        if id_ in ann:
            id_ = "{}_{}".format(id_, ann[id_])
        key_url_list.append((id_, url))
    return key_url_list


def DownloadImage(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '%s.png' % key)

    if os.path.exists(filename):
        # print('Image %s already exists. Skipping download.' % filename)
        return
    try:
        response = requests.get(url, timeout=10.0)
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    image_data = response.content
    if response.status_code != 200:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image %s %s' % (key, url))
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb.save(filename, format='PNG')
    except:
        print('Warning: Failed to save image %s' % filename)
        return


def Run():
    if len(sys.argv) != 3:
        print('Syntax: %s <train|validation|test.json> <output_dir/>' % sys.argv[0])
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = ParseData(data_file)
    pool = multiprocessing.Pool(processes=12)

    with tqdm(total=len(key_url_list)) as t:
        for _ in pool.imap_unordered(DownloadImage, key_url_list):
            t.update(1)


if __name__ == '__main__':
    Run()
