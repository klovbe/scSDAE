# scSDAE
Sparsity-penalized stacked denoising autoencoders for imputing single-cell RNA-seq data
# System requirements
+ Python 3.7. (version tested) or compatible
+ Tensorflow 1.12.0 (version tested) or compatible
+ keras
# Installation guide
Clone the github repository, install the dependencies 
# Usage for dataset with csv file with rows representing genes, columns representing cells
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_iters_ae', default=2000, type=int) # iteration steps for scSADE
    parser.add_argument('--n_iters_pretrain', default=1000, type=int) # iteration steps for scSADE
    parser.add_argument('--alpha', default=1.0, type=float)  # the mixture coefficient of mixture loss 
    parser.add_argument('--dr_rate', default=0.2, type=float) # dropout drate
    parser.add_argument('--nu1', default=0.0, type=float) # L1 regularization
    parser.add_argument('--nu2', default=0.0, type=float) # L2 regularization
    parser.add_argument("--train_datapath", default="./zeisel_count.csv", type=str) #path of the train file
    parser.add_argument("--data_type", default="count", type=str) 
    parser.add_argument("--outDir", default="./", type=str) #path to save output
    parser.add_argument("--name", default="zeisel", type=str)  #dataset name
    feature_parser = parser.add_mutually_exclusive_group(required=False) 
    feature_parser.add_argument('--gene_scale', dest='gene_scale', action='store_true')
    feature_parser.add_argument('--no-gene_scale', dest='gene_scale', action='store_false') 
    parser.set_defaults(gene_scale=False)  # if scale gene to [0,1]
    parser.add_argument('--GPU_SET', default="3", type=str) 
# Usage for folder containing csv files
    First, format the files into: {name}\_{data_type}.csv, then fill the blank in the file "run_pure_ae.csv":
    data_dir = "" # data path storing the data
    python_path = "" # path of python to use
    script_path = "{path}/pure_ae_new.csv"  # path storing pure_ae_new.csv
    run_log_path = "" # path to store the running logfile
    out_path = ""  # path to store the output
# output
    csv file "autoencoder_r.csv" with rows representing cells, columns representing genes
    
    

