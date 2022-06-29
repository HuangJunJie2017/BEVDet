# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['./bevdet-sttiny.py']

model = dict(
    img_view_transformer=dict(accelerate=True),
)