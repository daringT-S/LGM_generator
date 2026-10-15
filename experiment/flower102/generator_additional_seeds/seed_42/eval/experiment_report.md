# Generator seed 42: eval

状態: success

```json
{
  "status": "success",
  "completed_evaluations": [
    {
      "evaluation": 1,
      "seed": 3407,
      "top1": 0.9954464435577393,
      "top5": 0.9982110857963562
    },
    {
      "evaluation": 2,
      "seed": 3408,
      "top1": 0.9954464435577393,
      "top5": 0.9980484843254089
    },
    {
      "evaluation": 3,
      "seed": 3409,
      "top1": 0.99398273229599,
      "top5": 0.9983737468719482
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
  "data_path": "/gs/bs/tga-lab_otm/us07824-lgm-generator-seeds-wrpWhzXU/seed_42/eval/input/flowers102/dinov2_vitb/generator_seed42_final/data.pth",
  "data_sha256": "ba1ed540029844b879f7b7c6699c74a2c64c8588c900f076d0fef2025a1069bb",
  "std_ddof": 0,
  "job_id": "9011850",
  "host": "r11n8",
  "gpu": "NVIDIA H100",
  "torch": "2.8.0+cu126",
  "cuda_build": "12.6",
  "source_commit": "d0dbe18683ea2f0bccda7807ca9f61c73908bced",
  "current_evaluation": 3,
  "current_epoch": 900,
  "summary": {
    "top1_mean": 0.9949585398038229,
    "top1_std": 0.0006900001059213498,
    "top5_mean": 0.9982111056645712,
    "top5_std": 0.00013278787931978361
  },
  "evaluator": "official distillation.eval",
  "seeds": [
    3407,
    3408,
    3409
  ],
  "classifier_reinitialized_each_repeat": true,
  "pixel_top1_reference_percent": 97.1432,
  "top1_delta_percentage_points": 2.3526539803823,
  "total_seconds": 404.8883511950262,
  "peak_gpu_memory_bytes": 2197092352,
  "peak_reserved_gpu_memory_bytes": 2734686208
}
```

実験設定: `experiment/plan.json`。既存の実験を上書きしない。
