import torch
from typing import List, Tuple, Optional

def pad_sequence_list_to_batch(
    sequence_list: List[torch.Tensor],
    padding_value: int,
    max_len: Optional[int] = None,
    batch_first: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of variable-length 1D tensors to create a 2D batch tensor and its attention mask.

    Args:
        sequence_list: List of 1D tensors (sequences).
        padding_value: Value to use for padding.
        max_len: Maximum length to pad/truncate sequences to. 
                 If None, pads to the length of the longest sequence in the list.
        batch_first: If True, output tensor will be (batch_size, max_len). 
                     Otherwise (max_len, batch_size). (Currently only True is well-supported by example)
        device: The torch device for the output tensors. If None, uses the device
                of the first tensor in sequence_list or defaults to CPU.

    Returns:
        A tuple containing:
        - padded_sequences (torch.Tensor): The 2D tensor of padded sequences.
        - attention_mask (torch.Tensor): The 2D attention mask (1 for real tokens, 0 for padding).
    """
    if not sequence_list:
        # Return empty tensors on the specified device or CPU
        dev = device if device is not None else torch.device("cpu")
        return torch.empty((0, 0), dtype=torch.long, device=dev), torch.empty((0, 0), dtype=torch.long, device=dev)

    if device is None:
        device = sequence_list[0].device
    
    if max_len is None:
        max_len = max(seq.size(0) for seq in sequence_list)

    num_sequences = len(sequence_list)
    
    if batch_first:
        padded_sequences = torch.full((num_sequences, max_len), padding_value, dtype=torch.long, device=device)
        attention_mask = torch.zeros((num_sequences, max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(sequence_list):
            length = min(seq.size(0), max_len)
            padded_sequences[i, :length] = seq[:length]
            attention_mask[i, :length] = 1
    else: # Not thoroughly tested in current LLMGenerationManager context, but good to have
        padded_sequences = torch.full((max_len, num_sequences), padding_value, dtype=torch.long, device=device)
        attention_mask = torch.zeros((max_len, num_sequences), dtype=torch.long, device=device)
        for i, seq in enumerate(sequence_list):
            length = min(seq.size(0), max_len)
            padded_sequences[:length, i] = seq[:length]
            attention_mask[:length, i] = 1
            
    return padded_sequences, attention_mask
