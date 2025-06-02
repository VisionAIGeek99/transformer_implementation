import torch
from torch.utils.data import DataLoader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import prepare_device

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    txt_path = cfg.dataset_meta.raw_path
    context_length = cfg.dataset_meta.context_length
    batch_size = cfg.training.batch_size
    tokenizer = instantiate(cfg.tokenizer, text=txt_path)
    tokenizer.load(cfg.vocab.vocab_path)

    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset, path=txt_path, tokenizer=tokenizer, context_length=context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=cfg.training.shuffle)


    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(1)
    model = model.to(device)
    
    if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    for x, y in dataloader:
        x_decoded = tokenizer.decode(x[0].tolist())
        y_decoded = tokenizer.decode(y[0].tolist())
        print(f"x_decoded: {x_decoded}")
        print(f"y_decoded: {y_decoded}")

        break


if __name__ == "__main__":
    main()
