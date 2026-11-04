Model artifacts:
- leaflens_tomato.pt: trained MobileNetV3-Small checkpoint
- metrics.json: real evaluation results from held-out 75 images

Only trusted checkpoints should be loaded with PyTorch. The project uses torch.load(..., weights_only=True).
