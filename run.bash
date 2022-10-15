#!/bin/bash
# Adjusting [eval_batch_size] allows the display memory trades for evaluation speed. e.g. --eval_batch_size=1638200

# train HGCC, i.e. using interaction only
python run.py -mn hgcc -dn gowalla

# train HGCC+, i.e. using interaction and side information
# python run.py -mn hgcc -dn gowalla --spaces="{'inter': ['user', 'item'], 'net': ['user', 'user'], 'geon': ['item', 'item']}" --curvature="{'inter': 1., 'net': 1., 'geon': 1.}"