import tensorflow as tf

@tf.function
def normalise(image, new_min, new_max):
    """Normalises image to original bounds

    Args:
        image
        new_min: lower bound
        new_max: upper bound 

    Returns:
        normalised image
    """
    image = tf.cast(image, tf.float32)
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)
    scaled = (image - min_val) / (max_val - min_val + 1e-10)
    return scaled * (new_max - new_min) + new_min

@tf.function
def normalise_to_confidence_change(gradient_map, confidence_change):
    """Normalises gradient map such that the sum of the pixels equals the change in confidence.

    Args:
        gradient_map
        confidence_change: change in confdience between iteration k and k-1

    Returns:
        normalise gradient map
    """
    gradient_sum = tf.reduce_sum(gradient_map)
    scaling_factor = confidence_change / (gradient_sum + 1e-10)
    return scaling_factor * gradient_map

@tf.function
def compute_confidence(model, image, target_class, preprocess_fn):
    """ Preprocesses input according to preprocessing requirements detailed in 'preprocess_fn', passes it through the model and returns the model confidence for the target class

    Args:
        model
        image
        target_class: class of interest 
        preprocess_fn: the preprocessing required for model

    Returns:
        Confidence of model in the target class
    """
    image = tf.cast(image, tf.float32)
    image_scaled = normalise(image, 0.0, 255.0)
    image_processed = preprocess_fn(image_scaled)
    image_expanded = tf.expand_dims(image_processed, axis=0)
    predictions = model(image_expanded, training=False)
    return predictions[0, target_class]

@tf.function
def compute_gradients(model, image, target_class, preprocess_fn):
    """ Computes the model gradient of for each feature and returns a gradient map the same dimension as the input

    Args:
        model
        image
        target_class: class of interest 
        preprocess_fn: the preprocessing required for model

    Returns:
        Gradient map 
    """
    image = tf.cast(image, tf.float32)
    image_scaled = normalise(image, 0.0, 255.0)
    image_processed = preprocess_fn(image_scaled)
    image_expanded = tf.expand_dims(image_processed, axis=0)
    with tf.GradientTape() as tape:
        tape.watch(image_expanded)
        predictions = model(image_expanded, training=False)
        confidence = predictions[:, target_class]
    gradients = tape.gradient(confidence, image_expanded)
    return gradients[0]

@tf.function
def batched_confidences(model, images, target_class, preprocess_fn):
    """ Alternitive to 'compute confidence' function that will compute the confidences of perturbed inputs in a batch rather than individually

    Args:
        model
        perturbed images
        target_class: class of interest 
        preprocess_fn: the preprocessing required for model

    Returns:
        Confidences of perturbed images for target class  
    """
    images = tf.cast(images, tf.float32)
    images = tf.map_fn(lambda img: preprocess_fn(normalise(img, 0.0, 255.0)), images)
    predictions = model(images, training=False)
    return predictions[:, target_class]


@tf.function
def single_SIGMA_path(image, model, target_class, beta, alpha, epsilon, preprocess_fn):
    """Computes a SIGMA path, returning the attribution for that path 

    Args:
        image: image to be explained
        model: model to be explained
        target_class: class of interest
        beta: perturbation magnitude
        alpha: step size
        epsilon: stopping criteria
        preprocess_fn: preprocessing required for specified model

    Returns:
        final attribution map, the confidence of the model at each step in the path, the image at the end of the SIGMA path
    """
    shape = tf.shape(image)
    estimate_SIGMA = tf.identity(image)
    attribution = tf.zeros_like(image)

    original_confidence = compute_confidence(model, image, target_class, preprocess_fn)
    prev_confidence = original_confidence

    confidence_path = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    confidence_path = confidence_path.write(0, original_confidence)

    def cond_fn(estimate_SIGMA, attribution, prev_confidence, step, confidence_path):
        return tf.greater(prev_confidence, epsilon)

    def step_fn(estimate_SIGMA, attribution, prev_confidence, step, confidence_path):
        pattern = tf.cast(tf.random.uniform(shape, minval=0, maxval=2, dtype=tf.int32) * 255, tf.float32)
        increase = estimate_SIGMA + beta * pattern
        decrease = estimate_SIGMA - beta * pattern
        batched = tf.stack([increase, decrease], axis=0)
        confidences = batched_confidences(model, batched, target_class, preprocess_fn)
        perturbation_grad = (confidences[0] - confidences[1]) / (2.0 * beta)

        
        estimate_SIGMA = estimate_SIGMA - alpha * perturbation_grad * pattern
        estimate_SIGMA_norm = normalise(estimate_SIGMA, 0, 255)
        new_confidence = compute_confidence(model, estimate_SIGMA_norm, target_class, preprocess_fn)
        model_gradient = compute_gradients(model, estimate_SIGMA_norm, target_class, preprocess_fn)
        conf_diff = prev_confidence - new_confidence
        attribution += normalise_to_confidence_change(model_gradient, conf_diff)

        confidence_path = confidence_path.write(step + 1, new_confidence)
        return estimate_SIGMA, attribution, new_confidence, step + 1, confidence_path

    step = tf.constant(0)
    estimate_SIGMA, attribution, _, _, confidence_path = tf.while_loop(
        cond_fn, step_fn,
        [estimate_SIGMA, attribution, prev_confidence, step, confidence_path],
        maximum_iterations=1000
    )

    attribution_map = tf.reduce_sum(tf.abs(attribution), axis=-1)
    confidence_curve = confidence_path.stack()
    return attribution_map, confidence_curve, estimate_SIGMA



@tf.function
def single_SIGMA_path_adaptive(image, model, target_class,  alpha_range, beta, epsilon, preprocess_fn):
    """Computes a SIGMA path with adaptive alpha, returning the attribution for that path 

    Args:
        image: image to be explained
        model: model to be explained
        target_class: class of interest
        beta: perturbation magnitude
        epsilon: stopping criteria
        preprocess_fn: preprocessing required for specified model

    Returns:
        final attribution map, the confidence of the model at each step in the path, the image at the end of the SIGMA path
    """
    shape = tf.shape(image)
    estimate_SIGMA = tf.identity(image)
    attribution = tf.zeros_like(image)
    
    # for adaptive alpha range
    min_bound = tf.convert_to_tensor(alpha_range[0], dtype=tf.float32)
    max_bound = tf.convert_to_tensor(alpha_range[1], dtype=tf.float32)

    prev_estimate_SIGMA = tf.identity(image)
    prev_perturbation_grad = tf.zeros_like(image) 

    original_confidence = compute_confidence(model, image, target_class, preprocess_fn)
    prev_confidence = original_confidence

    confidence_path = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    confidence_path = confidence_path.write(0, original_confidence)

    def cond_fn(estimate_SIGMA, attribution, prev_confidence, step,
                confidence_path, prev_estimate_SIGMA, prev_perturbation_grad):
        return tf.greater(prev_confidence, epsilon)

    def step_fn(estimate_SIGMA, attribution, prev_confidence, step,
                confidence_path, prev_estimate_SIGMA, prev_perturbation_grad):

        pattern = tf.cast(tf.random.uniform(shape, minval=0, maxval=2, dtype=tf.int32) * 255, tf.float32)
        increase = estimate_SIGMA + beta * pattern
        decrease = estimate_SIGMA - beta * pattern
        batched = tf.stack([increase, decrease], axis=0)
        confidences = batched_confidences(model, batched, target_class, preprocess_fn)

        perturbation_grad = (confidences[0] - confidences[1]) / (2.0 * beta) * pattern

        delta_x = estimate_SIGMA - prev_estimate_SIGMA
        delta_g = perturbation_grad - prev_perturbation_grad
        numerator = tf.reduce_sum(delta_x * delta_x)
        denominator = tf.reduce_sum(delta_x * delta_g) + 1e-8

        adaptive_alpha = numerator/denominator
        
        adaptive_alpha_prime = tf.clip_by_value(adaptive_alpha, min_bound, max_bound)

        updated_estimate_SIGMA = estimate_SIGMA - adaptive_alpha_prime * perturbation_grad
        estimate_SIGMA_norm = normalise(updated_estimate_SIGMA, 0, 255)

        new_confidence = compute_confidence(model, estimate_SIGMA_norm, target_class, preprocess_fn)
        model_gradient = compute_gradients(model, estimate_SIGMA_norm, target_class, preprocess_fn)
        conf_diff = prev_confidence - new_confidence
        attribution += normalise_to_confidence_change(model_gradient, conf_diff)

        confidence_path = confidence_path.write(step + 1, new_confidence)

        return (
            updated_estimate_SIGMA,
            attribution,
            new_confidence,
            step + 1,
            confidence_path,
            estimate_SIGMA,
            perturbation_grad
        )

    estimate_SIGMA, attribution, prev_confidence, step, confidence_path, \
    prev_estimate_SIGMA, prev_perturbation_grad = tf.while_loop(
        cond_fn, step_fn,
        [estimate_SIGMA, attribution, prev_confidence, tf.constant(0),
         confidence_path, prev_estimate_SIGMA, prev_perturbation_grad],
        maximum_iterations=1000,
        shape_invariants=[
            image.shape, image.shape, tf.TensorShape([]), tf.TensorShape([]),
            tf.TensorShape(None), image.shape, image.shape
        ]
    )

    attribution_map = tf.reduce_sum(tf.abs(attribution), axis=-1)
    confidence_curve = confidence_path.stack()

    return attribution_map, confidence_curve, estimate_SIGMA
 


@tf.function
def SIGMA_attribution(model, image, target_class, n, beta, alpha, epsilon, preprocess_fn):
    """Computes the SIGMA attribution for n paths"

    Args:
        image: image to be explained
        model: model to be explained
        target_class: class of interest
        n: number of paths to average over 
        beta: perturbation magnitude
        alpha: step size
        epsilon: stopping criteria
        preprocess_fn: preprocessing required for specified model

    Returns:
        final averaged attribution map over n paths, the confidence of the model at each step in each path, the image at the end of the each SIGMA path
    """
    def parallel_body(_):
        return single_SIGMA_path(image, model, target_class, beta, alpha, epsilon, preprocess_fn)

    paths = tf.range(n)
    all_results = tf.map_fn(
        parallel_body,
        paths,
        fn_output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),    # attribution_map
            tf.RaggedTensorSpec(shape=[None], dtype=tf.float32),    # confidence_curve
            tf.TensorSpec(shape=image.shape, dtype=tf.float32),   # estimate_SIGMA
        )
    )

    all_maps = all_results[0]
    all_confidences = all_results[1]
    final_SIGMA_img = all_results[2]
    
    
    avg_attribution_map = tf.reduce_mean(all_maps, axis=0)
    return avg_attribution_map,all_confidences,final_SIGMA_img


@tf.function
def SIGMA_attribution_adaptive(model, image, target_class, n, alpha_range, beta, epsilon, preprocess_fn):
    """Computes the SIGMA attribution for n paths with adaptive alpha

    Args:
        image: image to be explained
        model: model to be explained
        target_class: class of interest
        n: number of paths to average over 
        alpha_range:(min_alpha, max_alpha) for adaptive alpha bounds
        beta: perturbation magnitude
        epsilon: stopping criteria
        preprocess_fn: preprocessing required for specified model

    Returns:
        final averaged attribution map over n paths, the confidence of the model at each step in each path, the image at the end of the each SIGMA path
    """
    def parallel_body(_):
        return single_SIGMA_path_adaptive(image, model, target_class, alpha_range, beta, epsilon, preprocess_fn)

    paths = tf.range(n)
    all_results = tf.map_fn(
        parallel_body,
        paths,
        fn_output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),    # attribution_map
            tf.RaggedTensorSpec(shape=[None], dtype=tf.float32),    # confidence_curve
            tf.TensorSpec(shape=image.shape, dtype=tf.float32),   # estimate_SIGMA
        )
    )

    all_maps = all_results[0]
    all_confidences = all_results[1]
    final_SIGMA_img = all_results[2]

    avg_attribution_map = tf.reduce_mean(all_maps, axis=0)
    return avg_attribution_map, all_confidences, final_SIGMA_img
