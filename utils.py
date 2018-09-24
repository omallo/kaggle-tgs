def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def moving_parameter_average(target_model, source_model, alpha):
    print("moving average: alpha=%.3f" % alpha)
    for param1, param2 in zip(target_model.parameters(), source_model.parameters()):
        print("moving average: param1_before=%.3f" % param1.data)
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
        print("moving average: param1_after=%.3f" % param1.data)
        print("moving average: param2=%.3f" % param2.data)
