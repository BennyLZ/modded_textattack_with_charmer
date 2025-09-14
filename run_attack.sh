#!/bin/bash
tar -xzf textattack_env.tar.gz
source textattack_env/bin/activate
python test_charmer_attack.py
