<!--  
 ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
 Han Cai, Ligeng Zhu, Song Han
 International Conference on Learning Representations (ICLR), 2019.
 -->
## Requirements
- Pytorch 1.3
- DALI 0.8
- pyyaml

```bash
# Train a model with DDP
python3 -u -m torch.distributed.launch --use_env --np
roc_per_node 8 imagenet_run_exp.py --path Exp/proxyless_mobile_14_dev --train --net proxyless_mobile_14 --dropout 0.3 --init_lr 0.3 --train_batch_size 96 --
n_worker 5

# Train a model: <path>/run.config; <path>/net.config
python python imagenet_run_exp.py --path <path> --train

# Train Proxyless (GPU)
python imagenet_run_exp.py --path Exp/proxyless_gpu --train --net proxyless_gpu --dropout 0.3

# Train Proxyless (CPU)
python imagenet_run_exp.py --path Exp/proxyless_cpu --train --net proxyless_cpu --dropout 0.2

# Train Proxyless (Mobile)
python imagenet_run_exp.py --path Exp/proxyless_mobile --train --net proxyless_mobile --dropout 0.1

# Train Proxyless (Mobile-14)
python imagenet_run_exp.py --path Exp/proxyless_mobile_14 --train --net proxyless_mobile_14 --dropout 0.3
```

```bash
# Eval Proxyless (GPU)
python imagenet_run_exp.py --path Exp/proxyless_gpu --net proxyless_gpu

# Eval Proxyless (CPU)
python imagenet_run_exp.py --path Exp/proxyless_cpu --net proxyless_cpu

# Eval Proxyless (Mobile)
python imagenet_run_exp.py --path Exp/proxyless_mobile --net proxyless_mobile

# Eval Proxyless (Mobile-14)
python imagenet_run_exp.py --path Exp/proxyless_mobile_14 --net proxyless_mobile_14
```

```bash
# Architecture Search
please refer to imagenet_arch_search.py
```