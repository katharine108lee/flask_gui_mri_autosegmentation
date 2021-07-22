
#!/bin/sh
module load anaconda

source /hpf/largeprojects/smiller/users/Katharine/python_environments/monai/bin/activate

python /hpf/largeprojects/smiller/users/Katharine/flask_project/predict/single_predict.py
