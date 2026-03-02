# Learning Accurate Segmentation Purely from Self-Supervision

<a href='https://arxiv.org/abs/2602.23759'><img src='https://img.shields.io/badge/ArXiv-Paper-red' /></a>
<a href='https://huggingface.co/geshang/Selfment'><img src='https://img.shields.io/badge/HuggingFace-Model-yellow' /></a>

https://github.com/user-attachments/assets/611641b4-22c9-4824-8a61-a5e5c73daf31

Introducing **Selfment**, a fully self-supervised framework that segments foreground objects directly from raw images without human labels, pretrained segmentation models, or any post-processing. 

---

## Todo

- [x] Release paper
- [x] DINOv3 PatchHead
- [x] Inference code
- [x] Multi-GPU eval code
- [ ] Multi-GPU training code



---

### Environment Setup

* We use python 3.11/CUDA 12.4/torch 2.9.1 for implementation.
* We train our models on 8 NVIDIA A100 GPUs with 80G memory, please make sure that your VRAM is sufficient to avoid the potential OOM issues during training.
* Download the [DINOv3-7B](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and the [PatchHead](https://huggingface.co/geshang/Selfment) into the `ckpt`, and download the required datasets ([COD10K](https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view), [CAMO](https://drive.google.com/file/d/1lLDZwQ0JiUM9FxTPGUGNQJhzBEkgm7x4/view?usp=sharing), [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip), [HKU-IS](https://pan.baidu.com/s/1c0EpNfM), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)) into the `datasets` folder.
* Install dependencies by:

```bash
pip install -r requirements.txt
```

---

### Quick Start

```bash
python demo/demo.py \
  --image demo/camouflaged.jpg \
  --head_ckpt /path/to/checkpoint_epoch3.pth \
  --dino_type dinov3 \
  --dino_repo ./dino/dinov3 \
  --dino_model_name dinov3_vit7b16 \
  --dino_weights /path/to/dinov3-weights.pth \
  --dino_depth 40 \
  --postprocess none \
  --output_dir ./demo_outputs
```



---


### Evaluation

Run evaluation:

```bash
python inference.py \
  --head_ckpt /path/to/checkpoint_epoch3.pth \
  --input_dir /path/to/images \
  --output_dir ./inference_results \
  --cache_dir ./feature_cache \
  --dino_type dinov3 \
  --dino_repo ./dino/dinov3 \
  --dino_model_name dinov3_vit7b16 \
  --dino_weights /path/to/dinov3-weights.pth \
  --dino_depth 40 \
  --postprocess none
```

And compute metrics:


```bash
python eval.py \
  --pred_dir /path/to/pred_masks \
  --gt_dir /path/to/gt_masks
```

---

## Citation

If you find our work helpful, please cite:

```bibtex
@article{you2026learning,
  title={Learning Accurate Segmentation Purely from Self-Supervision},
  author={You, Zuyao and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2602.23759},
  year={2026}
}
```

## Acknowledgements

Selfment is built upon [TokenCut](https://github.com/YangtaoWANG95/TokenCut) and [DINOv3](https://github.com/facebookresearch/dinov3/tree/main). We express our gratitude to the authors for their remarkable work.
