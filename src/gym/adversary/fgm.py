import tensorflow as tf
import numpy as np

def fgm(model_inputs,
        model_outputs,
        error_mult,
        eps=0.3,
        ord=np.inf):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param logits: output of model.get_logits
    :param y: (optional) A placeholder for the true labels. If targeted
                        is true, then provide the target label. Otherwise, only provide
                        this parameter if you'd like to use true labels when crafting
                        adversarial samples. Otherwise, model predictions are used as
                        labels to avoid the "label leaking" effect (explained in this
                        paper: https://arxiv.org/abs/1611.01236). Default is None.
                        Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                            Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                                     default, will try to make the label incorrect. Targeted
                                     will instead try to move in the direction of being more
                                     like y.
    :return: a tensor for the adversarial example
    """

    asserts = []

    # Compute loss
    no_backprop_model_outputs = tf.stop_gradient(model_outputs)
    bad_actions = tf.multiply(error_mult, no_backprop_model_outputs)
    #bad_actions = no_backprop_model_outputs
    loss = tf.losses.mean_squared_error(bad_actions, model_outputs)
    loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, model_inputs)

    optimal_perturbation = optimize_linear(grad, eps, ord)

    # Add perturbation to original example to obtain adversarial example
    adv_inputs = model_inputs + optimal_perturbation

    return adv_inputs, bad_actions

def optimize_linear(grad, eps, ord=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)
    :param grad: tf tensor containing a batch of gradients
    :param eps: float scalar specifying size of constraint region
    :param ord: int specifying order of norm
    :returns:
        tf tensor containing optimal perturbation
    """

    # In Python 2, the `list` call in the following line is redundant / harmless.
    # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
    red_ind = list(range(1, len(grad.get_shape())))
    avoid_zero_div = 1e-12
    if ord == np.inf:
        # Take sign of gradient
        optimal_perturbation = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `optimal_perturbation` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    elif ord == 1:
        abs_grad = tf.abs(grad)
        sign = tf.sign(grad)
        max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
        tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
        num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
    elif ord == 2:
        square = tf.maximum(avoid_zero_div,
                                                tf.reduce_sum(tf.square(grad),
                                                                     reduction_indices=red_ind,
                                                                     keepdims=True))
        optimal_perturbation = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                                            "currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = tf.multiply(eps, optimal_perturbation)
    return scaled_perturbation
