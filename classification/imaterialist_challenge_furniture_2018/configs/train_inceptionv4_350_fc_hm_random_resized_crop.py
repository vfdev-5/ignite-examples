# # Basic training configuration file
# import numpy as np
# import torch
# from torch.nn import functional as F
# from torch.optim import RMSprop
# from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
# from torch.utils.data.sampler import WeightedRandomSampler
# from torchvision.transforms import RandomHorizontalFlip
# from torchvision.transforms import RandomResizedCrop, RandomChoice
# from torchvision.transforms import ColorJitter, ToTensor, Normalize
# from ignite.engine import Events
# from common.dataset import FilesFromCsvDataset
# from common.data_loaders import get_data_loader
# from models.inceptionv4 import FurnitureInceptionV4_350_FC
#
#
# SEED = 42
# DEBUG = True
# DEVICE = 'cuda'
#
# OUTPUT_PATH = "output"
#
#
# size = 350
#
# TRAIN_TRANSFORMS = [
#     RandomChoice(
#         [
#             RandomResizedCrop(size, scale=(0.6, 8.0), interpolation=3),
#             RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=3),
#         ]
#     ),
#     RandomHorizontalFlip(p=0.5),
#     ColorJitter(hue=0.12, brightness=0.12),
#     ToTensor(),
#     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ]
#
# VAL_TRANSFORMS = [
#     RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=3),
#     RandomHorizontalFlip(p=0.5),
#     ToTensor(),
#     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ]
#
#
# BATCH_SIZE = 24
# NUM_WORKERS = 15
#
#
# dataset = FilesFromCsvDataset("output/filtered_train_dataset.csv")
# weights = [1.0] * len(dataset)
# train_sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
# TRAIN_LOADER = get_data_loader(dataset,
#                                data_transform=TRAIN_TRANSFORMS,
#                                batch_size=BATCH_SIZE,
#                                num_workers=NUM_WORKERS,
#                                pin_memory='cuda' in DEVICE)
#
#
# val_dataset = FilesFromCsvDataset("output/filtered_val_dataset.csv")
# VAL_LOADER = get_data_loader(val_dataset,
#                              data_transform=VAL_TRANSFORMS,
#                              batch_size=BATCH_SIZE,
#                              num_workers=NUM_WORKERS,
#                              pin_memory='cuda' in DEVICE)
#
#
# MODEL = FurnitureInceptionV4_350_FC(pretrained='imagenet')
#
#
# N_EPOCHS = 100
#
#
# OPTIM = RMSprop(
#     params=[
#         {"params": MODEL.stem.parameters(), 'lr': 0.0012},
#         {"params": MODEL.features.parameters(), 'lr': 0.0034},
#         {"params": MODEL.classifier.parameters(), 'lr': 0.0045},
#         {"params": MODEL.final_classifier.parameters(), 'lr': 0.045},
#     ],
#     alpha=0.9,
#     eps=1.0)
#
#
# LR_SCHEDULERS = [
#     MultiStepLR(OPTIM, milestones=[2, 4, 6, 8, 10, 12], gamma=0.94),
# ]
#
# # REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=2, threshold=0.1, verbose=True)
#
#
# EARLY_STOPPING_KWARGS = {
#     'patience': 15,
#     # 'score_function': None
# }
#
#
# new_weights = np.array(weights)
#
#
# def hnem_accumulate(evaluator, trainer, logger):
#
#     data_loader = evaluator.state.dataloader
#     output = evaluator.state.output
#
#     start_index = ((evaluator.state.iteration - 1) % len(data_loader)) * data_loader.batch_size
#     end_index = min(start_index + data_loader.batch_size, n_samples)
#     batch_y_probas = output['y_pred'].detach().numpy()
#     y_probas_tta[start_index:end_index, :, tta_index] = batch_y_probas
#     if tta_index == 0:
#         indices[start_index:end_index] = output['indices']
#
#
# def hnem_update(evaluator, trainer, logger):
#
#
#     assert hasattr(evaluator.state, "metrics"), "Evaluator state has no metrics"
#     recall_per_class = evaluator.state.metrics['recall'].cpu()
#     mean_recall_per_class = torch.mean(recall_per_class)
#     low_recall_classes = np.where(recall_per_class < 0.6 * mean_recall_per_class)[0]
#     logger.debug("Smart sampling update: low recall classes (< {}) : {}"
#                  .format(mean_recall_per_class, low_recall_classes))
#     n_classes = len(recall_per_class)
#     class_weights = []
#     for c in range(n_classes):
#         w = 5.0 if c in low_recall_classes else 1.0
#         class_weights.append((c, w))
#     train_sampler.weights =
#
#
# EVALUATOR_CUSTOM_EVENT_HANDLERS = [
#     # (event, handler), handler signature should be `foo(evaluator, trainer, logger)`
#     (Events.ITERATION_COMPLETED, hnem_accumulate)
#     (Events.COMPLETED, hnem_update)
# ]
#
# LOG_INTERVAL = 100
