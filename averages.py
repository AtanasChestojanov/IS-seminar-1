average_heuristic_authorship = 0
average_heuristic_friendship = 0
average_heuristic_row_wise = 0
average_heuristic_column_wise = 0
average_heuristic_block = 0
average_heuristic_swap = 0

average_matrix_authorship = 0
average_matrix_friendship = 0
average_matrix_row_wise = 0
average_matrix_column_wise = 0
average_matrix_block = 0
average_matrix_swap = 0

average_two_point_authorship = 0
average_two_point_friendship = 0
average_two_point_row_wise = 0
average_two_point_column_wise = 0
average_two_point_block = 0
average_two_point_swap = 0



for i in range(100):

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_heuristic,\
                                                                            mutation_type = custom_mutation_authorship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_heuristic_authorship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_heuristic,\
                                                                            mutation_type = custom_mutation_friendship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_heuristic_friendship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_heuristic,\
                                                                            mutation_type = custom_mutation_row_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_heuristic_row_wise += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_heuristic,\
                                                                            mutation_type = custom_mutation_column_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_heuristic_column_wise += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_heuristic,\
                                                                            mutation_type = custom_mutation_block, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_heuristic_block += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_heuristic,\
                                                                            mutation_type = custom_mutation_swap, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_heuristic_swap += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_matrix,\
                                                                            mutation_type = custom_mutation_authorship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_matrix_authorship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_matrix,\
                                                                            mutation_type = custom_mutation_friendship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_matrix_friendship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_matrix,\
                                                                            mutation_type = custom_mutation_row_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_matrix_row_wise += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_matrix,\
                                                                            mutation_type = custom_mutation_column_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_matrix_column_wise += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_matrix,\
                                                                            mutation_type = custom_mutation_block, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_matrix_block += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_matrix,\
                                                                            mutation_type = custom_mutation_swap, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_matrix_swap += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_two_point,\
                                                                            mutation_type = custom_mutation_authorship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_two_point_authorship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_two_point,\
                                                                            mutation_type = custom_mutation_friendship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_two_point_friendship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_two_point,\
                                                                            mutation_type = custom_mutation_row_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_two_point_row_wise += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_two_point,\
                                                                            mutation_type = custom_mutation_column_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_two_point_column_wise += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_two_point,\
                                                                            mutation_type = custom_mutation_block, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_two_point_block += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/hard_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_two_point,\
                                                                            mutation_type = custom_mutation_swap, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)
    average_two_point_swap += best_solution_fitness



average_heuristic_authorship = average_heuristic_authorship/100
average_heuristic_friendship = average_heuristic_friendship/100
average_heuristic_row_wise = average_heuristic_row_wise/100
average_heuristic_column_wise = average_heuristic_column_wise/100
average_heuristic_block = average_heuristic_block/100
average_heuristic_swap = average_heuristic_swap/100

average_matrix_authorship = average_matrix_authorship/100
average_matrix_friendship = average_matrix_friendship/100
average_matrix_row_wise = average_matrix_row_wise/100
average_matrix_column_wise = average_matrix_column_wise/100
average_matrix_block = average_matrix_block/100
average_matrix_swap = average_matrix_swap/100

average_two_point_authorship = average_two_point_authorship/100
average_two_point_friendship = average_two_point_friendship/100
average_two_point_row_wise = average_two_point_row_wise/100
average_two_point_column_wise = average_two_point_column_wise/100
average_two_point_block = average_two_point_block/100
average_two_point_swap = average_two_point_swap/100

print("\n--- Average Heuristic Crossover Values ---")
print(f"Authorship: {average_heuristic_authorship:.3f}")
print(f"Friendship: {average_heuristic_friendship:.3f}")
print(f"Row-wise: {average_heuristic_row_wise:.3f}")
print(f"Column-wise: {average_heuristic_column_wise:.3f}")
print(f"Block: {average_heuristic_block:.3f}")
print(f"Swap: {average_heuristic_swap:.2f}")

print("\n--- Average Matrix Crossover Values ---")
print(f"Authorship: {average_matrix_authorship:.3f}")
print(f"Friendship: {average_matrix_friendship:.3f}")
print(f"Row-wise: {average_matrix_row_wise:.3f}")
print(f"Column-wise: {average_matrix_column_wise:.3f}")
print(f"Block: {average_matrix_block:.3f}")
print(f"Swap: {average_matrix_swap:.3f}")

print("\n--- Average Two-Point Crossover Values ---")
print(f"Authorship: {average_two_point_authorship:.3f}")
print(f"Friendship: {average_two_point_friendship:.3f}")
print(f"Row-wise: {average_two_point_row_wise:.3f}")
print(f"Column-wise: {average_two_point_column_wise:.3f}")
print(f"Block: {average_two_point_block:.3f}")
print(f"Swap: {average_two_point_swap:.3f}")


