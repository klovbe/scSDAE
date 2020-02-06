import os
import codecs
import subprocess
from time import time
import sys

done_model_names = ["havrin", "manno", "mannomouse"]
# done_model_names = [ "camp", "chen", "havrin", "kolod", "macosko", "manno", "treutlein", "zeisel"]
data_dir = "/data/wlchi/data/filter_data"
python_path = "/data/wlchi/anaconda3/envs/tf-gpu/bin/python"
script_path = "/data/wlchi/python_project/SSDAE/sdae.py"
done_path = "/data/wlchi/python_project/SSDAE/done_train.txt"
date_now ="no_gene_scale_short"
run_log_path = "./sdae/{}/run_logs".format(date_now)
out_path = "./sdae/{}/out".format(date_now)


name_list = os.listdir(data_dir)
name_list_new = []
for file in name_list:
    if "count.csv" in file:
        name_list_new.append(file)
name_list = name_list_new
beta1_list = [0, 1]
GPU_SET = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

for i, file in enumerate(name_list[start:end]):
    path = os.path.join(data_dir, file)
    name = file.split(".")[0].split("_")[0]
    data_type = file.split(".")[0].split("_")[1]
    for beta1 in beta1_list:
        run_log_path_name = os.path.join(run_log_path, name)
        run_log_path_name = os.path.join(run_log_path_name, str(beta1))
        out_dir = os.path.join(out_path, name)
        out_dir = os.path.join(out_dir, str(beta1))
        if os.path.exists(run_log_path_name) is False:
            os.makedirs(run_log_path_name)
        if os.path.exists(out_dir) is False:
            os.makedirs(out_dir)
        if name in done_model_names:
          print("{} has been trained before".format(name))
          continue
        par_dict = {
        "python": python_path,
        "script_path": script_path,
        "model_name": name,
        "data_type" :data_type,
        "datapath":path,
        "outDir": out_dir,
        "run_log_path_name": run_log_path_name,
        "beta1": beta1,
        "gpu_set":GPU_SET
        }
        cmd = "{python} {script_path} --name={model_name}  --train_datapath={datapath} --data_type={data_type} " \
            " --outDir={outDir}  --beta1={beta1} --GPU_SET={gpu_set}" \
              " --n_iters_ae=1000 --n_iters_pretrain=500" \
              " > {run_log_path_name}/{model_name}.log 2>&1".format(
        **par_dict
            )
        print("running {}...".format(cmd))
        ret = subprocess.check_call(cmd, shell=True, cwd="/data/wlchi/python_project/SSDAE/")
        print(ret)