#!/usr/bin/env bash

srun --account=CS-TEACH-2023 --nodes=1 --partition=gpu --gres=gpu:1 --ntasks=1 --time=00:15:00 --pty /bin/bash
