<p align="center">
  <img src="contents/output.gif" alt="large" width="400">
  <img src="contents/cifar_best.gif" alt="large" width="200">
</p>


# D3PM CIFAR10 classification in pytorch  (based on Minimal Implementation of a D3PM)

 The loss function utilized is a combination of cross-entropy loss and KL divergence loss.
- Add noise to and denoise the classes of CIFAR-10, using ResNet50 backboen to extract image features as conditional inputs for denoising and generating labels.

- The time step \( t \) is a discrete integer, but a smooth and continuous representation is generated through frequency embedding (using sine and cosine functions), gradually transforming the labels from \( y_0 \) (one-hot) to \( y_T \) (uniform distribution).


- The loss function utilized is a combination of cross-entropy loss and KL divergence loss.

## Usage


CIFAR10 classification in pytorch
```bash
python d3pm_class.py
```

These are the codes for generating images from the original post(Minimal Implementation of a D3PM)
Following is completely self-contained example.

```bash
python d3pm_runner.py
```

Following uses dit.py, for CIFAR-10 dataset.
  
```bash
python d3pm_runner_cifar.py
```

## Requirements

Install torch, torchvision, pillow, tqdm

```bash
pip install torch torchvision pillow tqdm
```

## Citation

This implementation:

```bibtex
@misc{d3pm_pytorch,
  author={Simo Ryu},
  title={Minimal Implementation of a D3PM (Structured Denoising Diffusion Models in Discrete State-Spaces), in pytorch},
  year={2024},
  howpublished={\url{https://github.com/cloneofsimo/d3pm}}
}
```

Original Paper:

```bibtex
@article{austin2021structured,
  title={Structured denoising diffusion models in discrete state-spaces},
  author={Austin, Jacob and Johnson, Daniel D and Ho, Jonathan and Tarlow, Daniel and Van Den Berg, Rianne},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={17981--17993},
  year={2021}
}
```
