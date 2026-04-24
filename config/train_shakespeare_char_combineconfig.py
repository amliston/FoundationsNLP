# Experiment 5: Combining Configurations
# Change: as described above
# All other hyperparameters match baseline

out_dir = 'out-shakespeare-combineconfig'

eval_interval = 250
eval_iters = 200            
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'nanoGPT-assignment'
wandb_run_name = 'combineconfig'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64         # CHANGED: baseline was 256

n_layer = 2         # CHANGED: baseline was 6              
n_head = 6
n_embd = 384
dropout = 0.4    # CHANGED: baseline was 0.2

learning_rate = 5e-4       # CHANGED: baseline was 1e-3    
max_iters = 5000            
lr_decay_iters = 5000       
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
weight_decay = 1e-1

device = 'cuda' # change to 'cpu' for CPU
compile = False
