import torch
import torchvision
import fastai
from fastai.vision.all import *

path = "./data"
train_path = "./data/Training Data"
val_path = "./data/Validation Data"


# Loading training data

item_tfms = [Resize(224)]
batch_tfms = [DihedralItem(p=0.01), RandomResizedCrop(size=224, min_scale=0.06),
              Warp(magnitude=0.3), Brightness(max_lighting=0.5), Contrast(max_lighting=0.3), Saturation(max_lighting=0.3),
              Hue(max_hue=0.2), RandomErasing(p=0.3, sh=0.2, max_count=3, min_aspect=0.2)]

db = DataBlock(blocks=(ImageBlock, CategoryBlock),
               get_items=get_image_files,
               item_tfms=item_tfms,
               batch_tfms=batch_tfms, 
               get_y=parent_label)

dataloader = db.dataloaders(path)
dataloader.show_batch()


# First training pass

learner = vision_learner(dataloader, resnet50, metrics=accuracy, path=".")
early_stop = EarlyStoppingCallback(patience=10)
save_best_model = SaveModelCallback(fname='best_model_resnet50')
learner.fit_one_cycle(35, cbs=[early_stop, save_best_model])
learner.load('best_model_resnet50')


# Finding learning rate, second training pass

def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    model.lr_find()
    
    losses = np.array(model.recorder.losses)
    min_loss_index = np.argmin(losses)
    lrs = model.recorder.lrs
    
    return lrs[min_loss_index] / 10

learning_rate = find_appropriate_lr(learner)
learner.fit_one_cycle(40, lr_max=slice(learning_rate/10, learning_rate), cbs=[early_stop, save_best_model])


# Unfreezing model for finer training

learner.unfreeze()
learner.fit_one_cycle(40, lr_max=slice(learning_rate/10, learning_rate), cbs=[early_stop, save_best_model])
learner.load('best_model_resnet50')
learning_rate = find_appropriate_lr(learner)
learner.fit_one_cycle(30, lr_max=slice(learning_rate/10, learning_rate), cbs=[early_stop, save_best_model])
learner.load('best_model_resnet50')


# Saving PyTorch model as ONNX model

labels = learner.dls.vocab

pytorch_model = learner.model.eval()
softmax_layer = torch.nn.Softmax(dim=1)
normalisation_layer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

final_model = nn.Sequential(
    normalisation_layer,
    pytorch_model,
    softmax_layer
)

dummy = torch.randn([1, 3, 224, 224], device=next(final_model.parameters()).device)

torch.onnx.export(
    final_model,
    dummy,
    "models/landscape_model_resnet50.onnx",
    do_constant_folding=True,
    export_params=True,
    input_names=['input'],
    output_names=['output'])