from typing import Dict, Callable, Union, List

import torch
from torch.utils.data.dataloader import default_collate
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone


class EmotionRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the emotion recognition task.
    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': (B x T x F) tensor
            'input_lens': (B,) tensor
            'supervisions': (B x 1) dict with B-sized lists of emotion, speaker and gender
        }
    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features
    """

    def __init__(
        self,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        """
        EmotionRecognition IterableDataset constructor.

        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        """
        super().__init__()
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        validate(cuts)
        cuts = cuts.sort_by_duration()

        for tfnm in self.cut_transforms:
            cuts = tfnm(cuts)
        inputs, inputs_len = self.input_strategy(cuts)
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        for tfnm in self.input_transforms:
            inputs = tfnm(inputs)
        batch = {
            "inputs": inputs,
            "inputs_len": inputs_len,
            "supervisions": default_collate(
                [
                    {
                        "emotion": supervision.custom["emotion"],
                        "speaker": supervision.speaker,
                        "gender": supervision.gender,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }
        return batch
