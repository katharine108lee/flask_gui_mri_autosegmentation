#!/bin/bash
module load tmux
module load anaconda
source /hpf/largeprojects/smiller/users/Katharine/python_environments/monai/bin/activate
tmux new-session -d -s htop-session 'python flask_project.py';  # start new detached tmux session, run htop
tmux split-window;                             # split the detached tmux session
tmux split-window;                             # split the detached tmux session
