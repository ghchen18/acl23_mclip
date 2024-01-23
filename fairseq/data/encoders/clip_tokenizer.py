# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@register_tokenizer("clip", dataclass=FairseqDataclass)
class ClipTokenizer(object):
    def __init__(self, *unused):
        import clip
        self.tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    def encode(self, x: str) -> str:
        return self.tokenizer.encode(x)

    def decode(self, x: str) -> str:
        return self.tokenizer.decode(x)
