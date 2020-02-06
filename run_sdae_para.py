import subprocess
from time import time
import os
from multiprocessing import Pool
import sys
import glob
import time
import subprocess

def cmdfunc(**par_dict):
    cmd = "{python} {script_path} --name={model_name}  --train_datapath={datapath} --data_type={data_type} " \
          " --outDir={outDir}  --beta1={beta1} --GPU_SET={gpu_set}" \
          "  --n_iters_ae=2000 --n_iters_pretrain=1000" \
          "> {run_log_path_name}/{model_name}.log 2>&1"\
        .format(
        **par_dict
    )
    print("running {}...".format(cmd))
    # ret = subprocess.check_call(cmd, shell=True, cwd="/home/xysmlx/python_project/SSDAE")
    os.system(cmd)
    # print(ret)


# done_model_names = ["baron", "biase", "camp", "darmanis","deng", "goolam", "klein", "kolod","macosko", "patel",
#            "pollen", "shekhar", "treutlein", "usoskin", "yan", 'zeisel']
# done_model_names = ["zeiselercc","ziegenhain","melanoma","stoeckiusraw","melanomastupid"]
# done_model_names = ["zeisel", "ziegenhain"]
# , "CELseq2","DropSeq","MARSseq","SmartSeq","SmartSeq2"
done_model_names = ["havrin", "manno", "mannomouse"]
# done_model_names = [ "camp", "chen", "havrin", "kolod", "macosko", "manno", "treutlein", "zeisel"]
data_dir = "/data/wlchi/data/filter_data/down"
python_path = "/data/wlchi/anaconda3/envs/tf-gpu/bin/python"
script_path = "/data/wlchi/python_project/SSDAE/sdae.py"
done_path = "/data/wlchi/python_project/SSDAE/done_train.txt"
date_now ="no_gene_scale"
run_log_path = "./sdae/{}/run_logs".format(date_now)
out_path = "./sdae/{}/out".format(date_now)



name_list = os.listdir(data_dir)
# name_list = ["zeisel_count.csv"]
name_list_new = []
for file in name_list:
    if "log.csv" in file:
        name_list_new.append(file)
name_list = name_list_new
print(len(name_list_new))
# beta1_list = [0.05, 0.2, 0.5, 2]
# beta1_list = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]
# beta1_list = [ 0.5, 1, 2]
gpu_set = "0"
beta1_list = [0, 1]
start_time = time.time()
# pool is max parallel number of data to run
pool = Pool(processes=1)
for i, file in enumerate(name_list):
    path = os.path.join(data_dir, file)
    name = file.split(".")[0].split("_")[0]
    data_type = file.split(".")[0].split("_")[1]
    name = file.split(".")[0]
    data_type = "logcount"
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
        "gpu_set":gpu_set
        }

        time.sleep(2)
        pool.apply_async(cmdfunc, kwds={**par_dict})
pool.close()
pool.join()
print("all model cost ", time.time() - start_time)

