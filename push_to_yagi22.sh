#!/bin/bash

rsync -av --exclude=*__pycache__ --exclude=*runs --exclude=*results \
--exclude=log/* --exclude=save_dir/* \
../random-nas-combined administrator@yagi22:/home/administrator/blume/progs/
