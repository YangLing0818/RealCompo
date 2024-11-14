# ![Alt text](figs/image.png)

Official implementation of our **training-free** text-to-image framework - [RealCompo](https://arxiv.org/abs/2402.12908) (**NeurIPS 2024**).

>[**RealCompo: Balancing Realism and Compositionality Improves Text-to-Image Diffusion Models**](https://arxiv.org/abs/2402.12908)    
>[Xinchen Zhang\*](https://cominclip.github.io/), 
>[Ling Yang\*](https://yangling0818.github.io/), 
>[Yaqi Cai](https://github.com/CCYaqi),
>[Zhaochen Yu](https://github.com/BitCodingWalkin), 
>[Kai-Ni Wang](https://scholar.google.com.hk/citations?user=nMRUtZsAAAAJ&hl=zh-CN), 
>[Jiake Xie](https://scholar.google.com/citations?hl=zh-CN&user=pD8wUxgAAAAJ),
>[Ye Tian](https://github.com/tyfeld),
>[Minkai Xu](https://minkaixu.com/),
>[Yong Tang](https://scholar.google.com/citations?user=ubVWrRwAAAAJ&hl=en), 
>[Yujiu Yang](https://sites.google.com/view/iigroup-thu/about), 
>[Bin Cui](https://cuibinpku.github.io/) 
><br>**Tsinghua University, Peking University, University of Science and Technology of China, Southeast University, LibAI Lab, Stanford University**<br>

<details>
    <summary>Click for full abstract</summary>
Diffusion models have achieved remarkable advancements in text-to-image generation. However, existing models still have many difficulties when faced with multiple-object compositional generation. In this paper, we propose RealCompo, a new training-free and transferred-friendly text-to-image generation framework, which aims to leverage the respective advantages of text-to-image models and spatial-aware image diffusion models (e.g., layout, keypoints and segmentation maps) to enhance both realism and compositionality of the generated images. An intuitive and novel balancer is proposed to dynamically balance the strengths of the two models in denoising process, allowing plug-and-play use of any model without extra training. Extensive experiments show that our RealCompo consistently outperforms state-of-the-art text-to-image models and spatial-aware image diffusion models in multiple-object compositional generation while keeping satisfactory realism and compositionality of the generated images. Notably, our RealCompo can be seamlessly extended with a wide range of spatial-aware image diffusion models and stylized diffusion models.
</details>

## Introduction

![Alt text](figs/pipeline.png)

We introduce a new **training-free and transferred-friendly** text-to-image generation framework RealCompo that utilizes a novel balancer to achieve dynamic equilibrium between realism and compositionality in generated images.

## New Updates

**[2024.5]** Our main code of style-based and keypoint-based RealCompo is released.

**[2024.2]** Our main code of layout-based RealCompo is released.


## TODO
- [x] Release layout-based RealCompo
- [x] Release style-based RealCompo
- [x] Release keypoint-based RealCompo
- [ ] Release segmentation-based RealCompo

## Gallery

<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="figs/realcompo_layout.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Qualitative comparison between our RealCompo and the outstanding text-to-image model Stable Diffusion v1.5, as well as the layout-to-image models, GLIGEN and LMD+. Colored text denotes the advantages of RealCompo in generating results.
</td>
  </tr>
</table>

<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="figs/realcompo_key_seg.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Extend RealCompo to keypoint- and segmentation-based text-to-image generation.
</td>
  </tr>
</table>
<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="figs/realcompo_stylized.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Extend RealCompo to stylized compositional generation.
</td>
  </tr>
</table>
<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="figs/ablation.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Qualitative comparison of RealCompo's generalization to different models: We select two T2I models: Stable Diffusion v1.5, TokenCompose, two L2I models GLIGEN, Layout Guidance (LayGuide), and combine them in pairs to obtain four versions of RealCompo. We demonstrate that RealCompo has strong generalization and generality to different models, achieving a remarkable level of both fidelity and precision in aligning with text prompts.
</td>
  </tr>
</table>

## Installation

```shell
git clone https://github.com/YangLing0818/RealCompo
cd RealCompo
conda create -n RealCompo python==3.8.10
conda activate RealCompo
pip install -r requirements.txt
```

## Download Models

We provide the code of RealCompo v1, which is composed of Stable Diffusion v1.5 and GLIGEN.

You should download the checkpoints of GLIGEN ([HF Hub](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin)) put its path into  `inference_layout.py`.

## Generating images with Layout-based RealCompo

### Option 1: Use LLMs to reason out the layout

You can get the results through running: 

```bash
python inference_layout.py --user_prompt 'Two cute small corgi sitting in a movie theater with two popcorns in front of them.' --api_key 'put your api_key here' 
```

**--user_prompt** is the original prompt that used to generate a image.

**--api_key** is needed if you use GPT-4.

**You can also use local LLMs to reason out layouts**. Example samples will be saved in `generation_samples`. You can check `inference_layout.py` for more details about interface. 

```
generation_samples
├── generation_realcompo_v1_sd_gligen_two_cute_small_corgi_sitting_in_a_movie_theater_
│   ├── 0.png
│   ├── 1.png
|   .....
......
```

### Option 2: Manually setting the layout

If you already have the layouts related to all objects, you can directly run:

```bash
python inference_layout.py  --no_gpt --user_prompt 'Two cute small corgi sitting in a movie theater with two popcorns in front of them.' --object "['a cute small corgi', 'a cute small corgi', 'a movie theater', 'popcorn', 'popcorn']" --boundingbox "[[0.05, 0.05, 0.52, 0.58], [0.52, 0.05, 1.0, 0.58], [0.0, 0.0, 1, 1], [0.0, 0.6, 0.48, 0.95], [0.52, 0.6, 1, 0.95]]" --token_location "[4, 4, 9, 12, 12]"
```

**--no_gpt** can be used when you have already obtained the layout.

**--object** represents the set of objects mentioned in the prompt.

**--boundingbox** represents the set of layout for each object.

**--token_location** represents the set of locations where each object appears in the prompt.



You can change the backbone of the T2I model to Stable Diffusion v1.4, TokenCompose, and so on. 

The core code for updating the models' coefficients is located in `ldm/models/diffusion/plms.py`. Using this code, you can make slight modifications to replace the L2I model with another one.

## Generating images with Style-based RealCompo

You can use RealCompo to achieve stylized compositional generation by running:

```shell
 python inference_layout.py  --no_gpt --style 'coloring-pages' --user_prompt 'Coloring page of a car park in front of a house.' --object "['a car', 'a house']" --boundingbox "[[0.0, 0.5, 1.0, 0.9], [0.1, 0.0, 1.0, 0.6]]" --token_location "[5, 11]"
```

 **--style** represents the style you want for generation.

In this code, we provide two styles: 'coloring-pages' and 'cuteyukimix'. 

You can find more stylized T2I backbones in [Civitai](https://civitai.com/).

## Generating images with Keypoint-based RealCompo

```shell
python inference_keypoint.py --user_prompt 'Elsa and Anna, sparks of magic between them, princess dress, background with sparkles, black purple red color schemes.' --token_location "[1, 3]"
```

**--user_prompt** is the original prompt that used to generate a image.

**--token_location** represents the set of locations where each object appears in the prompt.

You can change the backbone of the T2I model to various SDXL-based models.

## Citation

```
@article{zhang2024realcompo,
  title={RealCompo: Balancing Realism and Compositionality Improves Text-to-Image Diffusion Models},
  author={Zhang, Xinchen and Yang, Ling and Cai, Yaqi and Yu, Zhaochen and Wang, Kaini and Xie, Jiake and Tian, Ye and Xu, Minkai and Tang, Yong and Yang, Yujiu and Cui, Bin},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
## Acknowledgements

This repo uses some codes from  [GLIGEN](https://github.com/gligen/GLIGEN) and [LLM-groundedDiffusion](https://github.com/TonyLianLong/LLM-groundedDiffusion). Thanks for their wonderful work and codebase! 
