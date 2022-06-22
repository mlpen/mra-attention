from timm.data import create_dataset, create_loader

def get_train_loader(batch_size, num_workers):
    dataset = create_dataset("", root = "datasets/imagenet", split = "train", is_training = True, batch_size = batch_size, repeats = 0.)
    loader = create_loader(
        dataset,
        input_size = (3, 224, 224),
        batch_size = batch_size,
        is_training = True,
        use_prefetcher = True,
        no_aug = False,
        re_prob = 0.0,
        re_mode = "pixel",
        re_count = 1,
        re_split = False,
        scale = [0.08, 1.0],
        ratio = [0.75, 1.3333333333333333],
        hflip = 0.5,
        vflip = 0.0,
        color_jitter = 0.4,
        auto_augment = "rand-m9-mstd0.5-inc1",
        num_aug_repeats = 0,
        num_aug_splits = 0,
        interpolation = "random",
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225),
        num_workers = num_workers,
        distributed = False,
        collate_fn = None,
        pin_memory = True,
        use_multi_epochs_loader = False,
        worker_seeding = "all"
    )
    return loader

def get_val_loader(batch_size, num_workers):
    dataset = create_dataset("", root = "datasets/imagenet", split = "validation", is_training = False, batch_size = batch_size)
    print(len(dataset) // batch_size)
    loader = create_loader(
        dataset,
        input_size = (3, 224, 224),
        batch_size = batch_size,
        is_training = False,
        use_prefetcher = True,
        interpolation = "bicubic",
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225),
        num_workers = num_workers,
        distributed = False,
        crop_pct = 0.95,
        pin_memory = True
    )
    return loader