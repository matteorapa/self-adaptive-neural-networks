import torch
from torchvision.models import resnet18
import torch_pruning as tp

if __name__ == '__main__':

    model = resnet18(pretrained=True)

    # Importance criteria
    example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.TaylorImportance()

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    iterative_steps = 5 # progressive pruning
    prune_amounts = [x / 64 for x in range(64)]
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity= 0.015625 , # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.TaylorImportance):
            # Taylor expansion requires gradients for importance estimation
            loss = model(example_inputs).sum() # a dummy loss for TaylorImportance
            loss.backward() # before pruner.step()
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("hist", pruner.pruning_history())
        # finetune your model here
        # finetune(model)
        # ...
        model