average_columns_authorship = 0
average_columns_friendship = 0
average_columns_rows = 0
average_columns_columns = 0
average_columns_block = 0
average_columns_swap = 0
average_rows_authorship = 0
average_rows_friendship = 0
average_rows_rows = 0
average_rows_columns = 0
average_rows_block = 0
average_rows_swap = 0
average_uniform_authorship = 0
average_uniform_friendship = 0
average_uniform_rows = 0
average_uniform_columns = 0
average_uniform_block = 0
average_uniform_swap = 0


for i in range(100):

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_columns,\
                                                                            mutation_type = custom_mutation_authorship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 100, \
                                                                            initial_solution = solution_capacity_min_reviews, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_authorship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_columns,\
                                                                            mutation_type = custom_mutation_friendship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_friendship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_columns,\
                                                                            mutation_type = custom_mutation_row_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_rows += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_columns,\
                                                                            mutation_type = custom_mutation_column_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_random, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_columns += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_columns,\
                                                                            mutation_type = custom_mutation_block, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_authorship_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_block += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_columns,\
                                                                            mutation_type = custom_mutation_swap, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_friendship_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_swap += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_rows,\
                                                                            mutation_type = custom_mutation_authorship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_reviewers_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_rows_authorship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_rows,\
                                                                            mutation_type = custom_mutation_friendship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_combined_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_rows_friendship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_rows,\
                                                                            mutation_type = custom_mutation_row_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_capacity_min_reviews, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_rows_rows += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_rows,\
                                                                            mutation_type = custom_mutation_column_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_randomized_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_rows_columns += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_rows,\
                                                                            mutation_type = custom_mutation_block, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_general, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_rows_block += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_rows,\
                                                                            mutation_type = custom_mutation_swap, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_random, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_rows_swap += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_uniform,\
                                                                            mutation_type = custom_mutation_authorship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_authorship_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_uniform_authorship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_uniform,\
                                                                            mutation_type = custom_mutation_friendship, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_friendship_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_friendship += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_uniform,\
                                                                            mutation_type = custom_mutation_row_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_reviewers_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_uniform_rows += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_uniform,\
                                                                            mutation_type = custom_mutation_column_wise, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_combined_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_columns_columns += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_uniform,\
                                                                            mutation_type = custom_mutation_block, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_reviewers_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_uniform_block += best_solution_fitness

    best_assignment, best_solution_fitness, ga_instance = find_solution_final(file_path = "datasets/medium_dataset_1.json", \
                                                                            parent_selection_type = "tournament",\
                                                                            crossover_type = custom_crossover_uniform,\
                                                                            mutation_type = custom_mutation_swap, \
                                                                            num_solutions = 150, \
                                                                            num_parents_mating = 50, \
                                                                            num_generations = 1, \
                                                                            initial_solution = solution_combined_constraint, \
                                                                            mutation_percent_genes = 3, \
                                                                            elitism = 0)

    average_uniform_swap += best_solution_fitness




average_columns_authorship = average_columns_authorship / 100
average_columns_friendship = average_columns_friendship / 100
average_columns_rows = average_columns_rows / 100
average_columns_columns = average_columns_columns / 100
average_columns_block = average_columns_block / 100
average_columns_swap = average_columns_swap / 100
average_rows_authorship = average_rows_authorship / 100
average_rows_friendship = average_rows_friendship / 100
average_rows_rows = average_rows_rows / 100
average_rows_columns = average_rows_columns / 100
average_rows_block = average_rows_block / 100
average_rows_swap = average_rows_swap / 100
average_uniform_authorship = average_uniform_authorship / 100
average_uniform_friendship = average_uniform_friendship / 100
average_uniform_rows = average_uniform_rows / 100
average_uniform_columns = average_uniform_columns / 100
average_uniform_block = average_uniform_block / 100
average_uniform_swap = average_uniform_swap / 100


print("Average solution fitness - solution_capacity_min_reviews: ", average_columns_authorship)
print("Average solution fitness - solution_randomized_general: ", average_columns_friendship)
print("Average solution fitness - solution_general: ", average_columns_rows)
print("Average solution fitness - solution_random: ", average_columns_columns)
print("Average solution fitness - solution_authorship_constraint: ", average_columns_block)
print("Average solution fitness - solution_friendship_constraint: ", average_columns_swap)
print("Average solution fitness - solution_reviewers_constraint: ", average_rows_authorship)
print("Average solution fitness - solution_combined_constraint: ", average_rows_friendship)
print("Average solution fitness - solution_capacity_min_reviews: ", average_rows_rows)
print("Average solution fitness - solution_randomized_general: ", average_rows_columns)
print("Average solution fitness - solution_general: ", average_rows_block)
print("Average solution fitness - solution_random: ", average_rows_swap)
print("Average solution fitness - solution_authorship_constraint: ", average_uniform_authorship)
print("Average solution fitness - solution_friendship_constraint: ", average_uniform_friendship)
print("Average solution fitness - solution_reviewers_constraint: ", average_uniform_rows)
print("Average solution fitness - solution_combined_constraint: ", average_uniform_columns)
print("Average solution fitness - solution_capacity_min_reviews: ", average_uniform_block)
print("Average solution fitness - solution_randomized_general: ", average_uniform_swap)
