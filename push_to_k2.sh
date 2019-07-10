#!/bin/bash

rsync -av --exclude=*__pycache__ --exclude=*runs --exclude=*results ../random-res blume@k2:/ceph/blume/progs/
