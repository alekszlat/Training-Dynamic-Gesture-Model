import numpy as np
from dataclasses import dataclass


@dataclass
class NormalizedData:
    norm_sequence: np.ndarray
    mask: np.ndarray


class TemporalNormalizer:
    def __init__(self, target_length: int = 40):
        if target_length <= 0:
            raise ValueError("target_length must be greater than 0")

        self.target_length = target_length

    def normalize(self, sequence: np.ndarray) -> NormalizedData:
        """
        Normalize a landmark sequence to fixed length.

        Input:
            sequence: [T, 21, 3]

        Output:
            norm_sequence: [target_length, 21, 3]
            mask: [target_length]
        """

        original_length = sequence.shape[0]

        if original_length == 0:
            norm_sequence = np.zeros(
                (self.target_length, *sequence.shape[1:]),
                dtype=sequence.dtype,
            )
            mask = np.zeros(self.target_length, dtype=np.float32)

            return NormalizedData(norm_sequence=norm_sequence, mask=mask)

        if original_length == self.target_length:
            norm_sequence = sequence
            mask = np.ones(self.target_length, dtype=np.float32)
            return NormalizedData(norm_sequence=norm_sequence, mask=mask)
        elif original_length > self.target_length:
            # Uniformly sample frames across the whole sequence
            indices = np.linspace(
                0,
                original_length - 1,
                self.target_length,
            ).astype(int)

            norm_sequence = sequence[indices]
            mask = np.ones(self.target_length, dtype=np.float32)

        else:
            # Pad short sequence with zero frames
            norm_sequence = np.zeros(
                (self.target_length, *sequence.shape[1:]),
                dtype=sequence.dtype,
            )
            norm_sequence[:original_length] = sequence

            mask = np.zeros(self.target_length, dtype=np.float32)
            mask[:original_length] = 1.0

        return NormalizedData(norm_sequence=norm_sequence, mask=mask)

