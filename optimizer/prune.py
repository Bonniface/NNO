

# Store accuracies for plotting
global_best_accuracies = []

# Main optimization loop
while function_evaluations < max_function_evaluations:
  for iteration in range(maxIterations):
    for i in range(num_flies):
        # Dispersal: Randomly move some flies
        if np.random.rand() < 0.1:  # Dispersal probability
            flies[i] = np.random.uniform([pruning_ratio_bounds[0], quantization_precision_bounds[0]],
                                         [pruning_ratio_bounds[1], quantization_precision_bounds[1]])
        # Attraction: Move flies towards the best-known positions
        else:
            flies[i] += 0.1 * (global_best_position - flies[i])  # Attraction to global best

        # Ensure flies stay within bounds
        flies[i] = np.clip(flies[i], [pruning_ratio_bounds[0], quantization_precision_bounds[0]],
                           [pruning_ratio_bounds[1], quantization_precision_bounds[1]])

        # Evaluate new position
        accuracy = evaluate_pruned_quantized_model(flies[i][0], flies[i][1])
        function_evaluations += 1
        if accuracy > best_accuracies[i]:
            best_accuracies[i] = accuracy
            best_positions[i] = flies[i]
        if accuracy > global_best_accuracy:
            global_best_accuracy = accuracy
            global_best_position = flies[i]

    global_best_accuracies.append(global_best_accuracy)

print(f"Best pruning ratio: {global_best_position[0]}")
print(f"Best quantization precision: {global_best_position[1]}")
print(f"Best accuracy: {global_best_accuracy}")