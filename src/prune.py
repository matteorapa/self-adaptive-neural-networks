from utils import *
import torch.nn.utils.prune as prune
from torchvision.models import ResNet


def get_external_requirements():
    return 0.1


def apply_prune(model: ResNet, prune_amount: int) -> ResNet:
    for name, module in model.named_modules():
        # prune connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=prune_amount, n=2, dim=0)
            # prune connections in all linear layers
            # elif isinstance(module, torch.nn.Linear):
            #     prune.ln_structured(module, name="weight", amount=prune_amount, n=2, dim=0)

            prune.remove(module, 'weight')

    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    return model


def apply_channel_prune(model, sparsity, example_inputs):
        print("=> Applying pruning: '{}'".format(sparsity))
        # Importance criteria
        imp = tp.importance.TaylorImportance()

        ignored_layers = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)  # DO NOT prune the final classifier!

        iterative_steps = 1  # progressive pruning
        current_step = 1
        prune_amounts = [x / 64 for x in range(48)]

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            ch_sparsity=sparsity,
            ignored_layers=ignored_layers,
        )

        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

        for i in range(iterative_steps):
            if isinstance(imp, tp.importance.TaylorImportance):
                # Taylor expansion requires gradients for importance estimation
                loss = model(example_inputs).sum()  # a dummy loss for TaylorImportance
                loss.backward()  # before pruner.step()
            pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
            print("Pruning step:", current_step, "multiply–accumulate (macs):", macs, "number of parameters", nparams)
            current_step += 1
            print(
                "  Iter %d/%d, Params: %.2f M => %.2f M"
                % (i + 1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
            )
            print(
                "  Iter %d/%d, MACs: %.2f G => %.2f G"
                % (i + 1, iterative_steps, base_macs / 1e9, macs / 1e9)
            )


        # model_statistics = summary(model, (1, 3, 224, 224), depth=3,
        #                            col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"], )
        # model_statistics_str = str(model_statistics)

        history = pruner.pruning_history()
        layers_affected = len(history)
        layers_affected_per_step = int(layers_affected / iterative_steps)
        step_history = [history[i:i + layers_affected_per_step] for i in
                        range(0, layers_affected, layers_affected_per_step)]

        return model, step_history



