#!/bin/bash

rsync -av --exclude=*__pycache__ --exclude=*runs --exclude=*results ../random-nas-combined blume@k2:/home/blume/progs/
