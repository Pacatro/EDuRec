import torch


def collate_fn(batch: list[list[dict]]) -> dict[str, torch.Tensor]:
    flattened_batch = [item for sublist in batch for item in sublist]
    result = {}
    for key in flattened_batch[0].keys():
        tensors = [d[key] for d in flattened_batch]
        if tensors[0].dtype == torch.float32:
            result[key] = torch.stack(tensors).float()
        else:
            result[key] = torch.stack(tensors).long()
    return result
