"""File that contains the adapted aggregators in FLEX-NLP for fast
development of a federated model using the FLEXible environment.

This aggregators also can work as examples for creating a custom aggregator.
"""

import numpy as np
import tensorly as tl
from flex.pool.decorators import aggregate_weights
from flex.pool.decorators import set_tensorly_backend


def clip_avg_f(aggregate_weights_as_list: list, clip_threshold: float = 0.9):
    n_layers = len(aggregate_weights_as_list[0])
    agg_weights = []
    for layer_index in range(n_layers):
        weights_per_layer = []
        for client_weights in aggregate_weights_as_list:
            w = tl.tensor(client_weights[layer_index])
            weights_per_layer.append(w)
        weights_per_layer = tl.stack(weights_per_layer)
        clip_threshold = tl.quantile(weights_per_layer, clip_threshold)
        clipped_layer = tl.sum(tl.clip(weights_per_layer, -clip_threshold, clip_threshold), axis=0)
        agg_weights.append(clipped_layer)
    return agg_weights

@aggregate_weights
def clip_avg(aggregate_weights_as_list: list, clip_threshold: float = 0.9):
    """Aggregate the weights using the clip average method.
    This function calculates the quantile of the weights of each layer and
    then clips the weights to the interval [-quantile, quantile].

    Args:
        aggregate_weights_as_list (list): List of weights to aggregate.
        clip_threshold (float, optional): Quantile threshold to apply to each
        layer. Defaults to 0.9.

    Returns:
        list: List of aggregated weights.
    
    Example of use assuming you are using a client-server architecture:
        from flex.pool.primitives import clip_avg

        aggregator = flex.pool.aggregators
        server = flex.pool.servers
        clip_threshold = 0.98 # quantile to clip the weights
        aggregator.map(server, clip_avg, clip_threshold)

    Example of use using the FlePool without separating server
    and aggregator, and following a client-server architecture:

        from flex.pool.primitives import clip_avg
        clip_threshold = 0.98 # quantile to clip the weights
        flex_pool.aggregators.map(flex_pool.servers, clip_avg, clip_threshold=clip_threshold)    
    """
    set_tensorly_backend()
    return clip_avg_f(aggregate_weights_as_list, clip_threshold)
