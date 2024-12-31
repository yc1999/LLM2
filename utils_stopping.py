import numpy as np
import torch
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)

class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list):
        self.token_id_list = token_id_list
        self.stop_tag = None

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for _ in range(len(self.token_id_list)):
            stop_states = [
                np.array_equal(
                    self.token_id_list[_],
                    input_ids[i][-len(self.token_id_list[_]) :].detach().cpu().numpy(),
                )
                for i in range(input_ids.size(0))
            ]
            if self.stop_tag is None:
                self.stop_tag = stop_states
            else:
                self.stop_tag = [
                    self.stop_tag[i] or stop_states[i]
                    for i in range(len(self.stop_tag))
                ]
            if all(self.stop_tag):
                self.stop_tag = None
                return True
        return False