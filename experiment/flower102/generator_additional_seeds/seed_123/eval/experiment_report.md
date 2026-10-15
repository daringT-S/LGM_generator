# Generator seed 123: eval

状態: success

```json
{
  "status": "success",
  "completed_evaluations": [
    {
      "evaluation": 1,
      "seed": 3407,
      "top1": 0.9936575293540955,
      "top5": 0.9993494749069214
    },
    {
      "evaluation": 2,
      "seed": 3408,
      "top1": 0.9949585199356079,
      "top5": 0.9995121359825134
    },
    {
      "evaluation": 3,
      "seed": 3409,
      "top1": 0.9947959184646606,
      "top5": 0.9993494749069214
    }
  ],
  "config": {
    "dataset": "flowers102",
    "model": "dinov2_vitb",
    "eval_model": "dinov2_vitb",
    "num_eval": 3,
    "eval_epochs": 1000,
    "eval_it": -1,
    "patience": 5,
    "checkpoint_it": 100,
    "num_workers": 2,
    "real_batch_size": 100,
    "real_res": 256,
    "crop_res": 224,
    "train_crop_mode": "random"
  },
  "data_path": "/gs/bs/tga-lab_otm/us07824-lgm-generator-seeds-wrpWhzXU/seed_123/eval/input/flowers102/dinov2_vitb/generator_seed123_final/data.pth",
  "data_sha256": "f3dfe864f00636db49927cee738fe32e25d692360130dddfcf5e7d33b8a36a6f",
  "std_ddof": 0,
  "job_id": "9012260",
  "host": "r20n1",
  "gpu": "NVIDIA H100",
  "torch": "2.8.0+cu126",
  "cuda_build": "12.6",
  "source_commit": "d0dbe18683ea2f0bccda7807ca9f61c73908bced",
  "current_evaluation": 3,
  "current_epoch": 900,
  "summary": {
    "top1_mean": 0.9944706559181213,
    "top1_std": 0.000578786613777451,
    "top5_mean": 0.999403695265452,
    "top5_std": 7.667916639081987e-05
  },
  "evaluator": "official distillation.eval",
  "seeds": [
    3407,
    3408,
    3409
  ],
  "classifier_reinitialized_each_repeat": true,
  "pixel_top1_reference_percent": 97.1432,
  "top1_delta_percentage_points": 2.3038655918121407,
  "total_seconds": 412.3515640338883,
  "peak_gpu_memory_bytes": 2197092352,
  "peak_reserved_gpu_memory_bytes": 2734686208
}
```

実験設定: `experiment/plan.json`。既存の実験を上書きしない。
