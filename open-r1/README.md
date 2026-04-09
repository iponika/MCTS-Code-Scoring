The code for replicate R1 on our seed data:

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=1 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-7B-Instruct/grpo/config_demo.yaml
```

The code is adapted from the [open-r1 project](https://github.com/huggingface/open-r1).
