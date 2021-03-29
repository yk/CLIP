import numpy as np
import pytest
import torch
from PIL import Image

import clip


def test_load():
    device = "cpu"
    model, transform = clip.load('ViT-H/14', device=device)
    print(model)
