def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def moving_parameter_average(target_model, source_model, alpha):
    for param1, param2 in zip(target_model.parameters(), source_model.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha
