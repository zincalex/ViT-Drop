import logging

logger = logging.getLogger(__name__)

"""For recording weights"""
class HiddenStatesRecordWrapper:
    def __init__(self, layer, layer_name="none", record_input=True, record_output=True):
        self.layer = layer
        self.layer_name = layer_name
        self.record_input = record_input
        self.record_output = record_output

        if record_input:
            self.input_hidden_states = []
        if record_output:
            self.output_hidden_states = []

    def record(self, input, output):
        if self.record_input and input is not None:
            # Ensure input is 3D: (batch, seq_len, hidden_size)
            if input.dim() == 2:
                input = input.unsqueeze(0)
            self.input_hidden_states.append(input.squeeze(0).clone().cpu())
        if self.record_output and output is not None:
            # Ensure output is 3D: (batch, seq_len, hidden_size)  
            if output.dim() == 2:
                output = output.unsqueeze(0)
            self.output_hidden_states.append(output.squeeze(0).clone().cpu())
