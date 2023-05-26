def hello():
    print("It works!")


def get_external_requirements():
    # the level of adaptability needed
    return 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, labels):
    index = 0
    correct = 0
    total = len(output)

    for o in output:
        if o == labels[index]:
            correct += 1
        index += 1

    top1_accuracy = correct / total * 100
    return top1_accuracy