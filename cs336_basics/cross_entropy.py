from jaxtyping import Float, Int
from torch import Tensor, exp, log

def cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    max_logits = inputs.max(dim=1, keepdim=True).values # [batch_size, 1]
    inputs_stable = inputs - max_logits # [batch_size, vocab_size]

    target_logits = inputs_stable.gather(dim=1, index=targets.unsqueeze(dim=1))

    exp_inputs = exp(inputs_stable)
    sum_exp = exp_inputs.sum(dim=1, keepdim=True)
    log_sum_exp = log(sum_exp)

    sample_losses = -target_logits + log_sum_exp

    avg_loss = sample_losses.flatten().mean()

    return avg_loss

