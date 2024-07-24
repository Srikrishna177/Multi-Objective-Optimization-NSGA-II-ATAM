import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import time
from time import sleep

class DingNetProblem(Problem):
    def __init__(self, architecture):
        self.architecture = architecture
        self.possible_categories_reference_distance = [10, 100, 1000]
        self.possible_categories_bandwidth_transmission = [125, 250, 500]
        self.possible_categories_coding_rate = [0.5, 0.6, 0.7, 0.8]
        self.possible_categories_number_of_rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 250, 500, 1000]

        super().__init__(
            n_var=24, n_obj=2, n_constr=0,
            xl=np.array([1, 1, -3, 0.1, 1, 1, 7, 1, -20, 1, 0, 0, 7, 0, 0, 8, 1, -120, 1, 0, 0, -200, -200, 3]),
            xu=np.array([21, 1000, 14, 1000, 20, 21, 12, 20, 20, 4,
                         len(self.possible_categories_reference_distance) - 1,
                         len(self.possible_categories_bandwidth_transmission) - 1, 12, 125,
                         len(self.possible_categories_coding_rate) - 1,
                         12.25, 13, -30, 295, len(self.possible_categories_number_of_rounds) - 1, 1, 200, 200, 50]))

    def _evaluate(self, X, out, *args, **kwargs):

        energy_consumption = []
        reliability = []
        #g1 = []
        #g2 = []

        for x in X:
            (power_setting_mote, sampling_rate_mote, spreading_factor_mote, movement_speed_mote,
             transmission_power_threshold_mote, power_setting_gateway, spreading_factor_gateway,
             transmission_power_threshold_gateway, shadow_fading, path_loss_exponent, reference_distance_encoded,
             bandwidth_transmission_encoded, spreading_factor_transmission, transmission_power_transmission,
             coding_rate_encoded, payload_size, payload_length, signal_strength, shortest_distance,
             number_of_rounds_encoded, activity_probability, upper_bound, lower_bound, lorawan_data_rate) = x

            reference_distance = self.possible_categories_reference_distance[int(reference_distance_encoded)]
            bandwidth_transmission = self.possible_categories_bandwidth_transmission[int(bandwidth_transmission_encoded)]
            coding_rate = self.possible_categories_coding_rate[int(coding_rate_encoded)]
            number_of_rounds = self.possible_categories_number_of_rounds[int(number_of_rounds_encoded)]

            energy, reliability_val = self.simulate(
                power_setting_mote, sampling_rate_mote, spreading_factor_mote, movement_speed_mote,
                transmission_power_threshold_mote,
                power_setting_gateway, spreading_factor_gateway, transmission_power_threshold_gateway,
                shadow_fading, path_loss_exponent, reference_distance, bandwidth_transmission,
                spreading_factor_transmission, transmission_power_transmission, coding_rate,
                payload_size, payload_length, signal_strength, shortest_distance, number_of_rounds,
                activity_probability, upper_bound, lower_bound, lorawan_data_rate)

            energy_consumption.append(energy)
            reliability.append(reliability_val)
            #g1.append(lower_bound - upper_bound)
            #g2.append(reliability_val - 0.9)

        out["F"] = np.column_stack([energy_consumption, reliability])
        #out["G"] = np.column_stack([g1])

    def simulate(self, power_setting_mote, sampling_rate_mote, spreading_factor_mote, movement_speed_mote,
                 transmission_power_threshold_mote, power_setting_gateway, spreading_factor_gateway,
                 transmission_power_threshold_gateway, shadow_fading, path_loss_exponent, reference_distance,
                 bandwidth_transmission, spreading_factor_transmission, transmission_power_transmission, coding_rate,
                 payload_size, payload_length, signal_strength, shortest_distance, number_of_rounds,
                 activity_probability, upper_bound, lower_bound, lorawan_data_rate):

        if self.architecture == 'Component-Based':
            reusability_factor = 0.15
            loosely_coupling_factor = 0.05
            energy = (power_setting_mote * 0.73 + sampling_rate_mote * 0.64 + spreading_factor_mote * 0.47 +
                      movement_speed_mote * 0.35 + transmission_power_threshold_mote * 0.1 + power_setting_gateway * 0.9 +
                      spreading_factor_gateway * 0.6 + transmission_power_threshold_gateway * 0.3 + shadow_fading * 0.8 +
                      path_loss_exponent * 0.65 + reference_distance * 0.33 + bandwidth_transmission * 0.58 +
                      spreading_factor_transmission * 0.69 + transmission_power_transmission * 0.91 + coding_rate * 0.15 +
                      payload_size * 0.025 + payload_length * 0.025 + signal_strength * 0.6 + shortest_distance * 0.4 +
                      number_of_rounds * 0.35 + activity_probability * 0.65 + upper_bound * 0.7 + lower_bound * 0.55 + lorawan_data_rate * 0.65) * 10
            energy = energy * (1 - reusability_factor + loosely_coupling_factor)
            reliability = 1 / (1 + power_setting_mote * 0.77 + sampling_rate_mote * 0.45 + spreading_factor_mote * 0.68 +
                               movement_speed_mote * 0.25 + transmission_power_threshold_mote * 0.60 +
                               power_setting_gateway * 0.85 + spreading_factor_gateway * 0.8 +
                               transmission_power_threshold_gateway * 0.3 + shadow_fading * 0.87 + path_loss_exponent * 0.73 +
                               reference_distance * 0.36 + bandwidth_transmission * 0.55 + spreading_factor_transmission * 0.65 +
                               transmission_power_transmission * 0.7 + coding_rate * 0.35 + payload_size * 0.05 +
                               payload_length * 0.05 + signal_strength * 0.7 + shortest_distance * 0.3 +
                               number_of_rounds * 0.75 + activity_probability * 0.55 + upper_bound * 0.89 + lower_bound * 0.46 + lorawan_data_rate * 0.555)
            reliability = reliability * (1 + reusability_factor - loosely_coupling_factor) / 10
            final_reliability = reliability * 1e6

        elif self.architecture == 'Layered':
            separation_factor = 0.25
            interaction_overhead = 0.1
            energy = (power_setting_mote * 0.73 + sampling_rate_mote * 0.64 + spreading_factor_mote * 0.47 +
                      movement_speed_mote * 0.35 + transmission_power_threshold_mote * 0.1 + power_setting_gateway * 0.9 +
                      spreading_factor_gateway * 0.6 + transmission_power_threshold_gateway * 0.3 + shadow_fading * 0.8 +
                      path_loss_exponent * 0.65 + reference_distance * 0.33 + bandwidth_transmission * 0.58 +
                      spreading_factor_transmission * 0.69 + transmission_power_transmission * 0.91 + coding_rate * 0.15 +
                      payload_size * 0.025 + payload_length * 0.025 + signal_strength * 0.6 + shortest_distance * 0.4 +
                      number_of_rounds * 0.35 + activity_probability * 0.65 + upper_bound * 0.6 + lower_bound * 0.4 + lorawan_data_rate * 0.65) * 12
            energy = energy * (1 + interaction_overhead - separation_factor)
            reliability = 1 / (1 + power_setting_mote * 0.77 + sampling_rate_mote * 0.45 + spreading_factor_mote * 0.68 +
                               movement_speed_mote * 0.25 + transmission_power_threshold_mote * 0.60 +
                               power_setting_gateway * 0.85 + spreading_factor_gateway * 0.8 +
                               transmission_power_threshold_gateway * 0.3 + shadow_fading * 0.87 + path_loss_exponent * 0.73 +
                               reference_distance * 0.36 + bandwidth_transmission * 0.55 + spreading_factor_transmission * 0.65 +
                               transmission_power_transmission * 0.7 + coding_rate * 0.35 + payload_size * 0.05 +
                               payload_length * 0.05 + signal_strength * 0.7 + shortest_distance * 0.3 +
                               number_of_rounds * 0.75 + activity_probability * 0.55 + upper_bound * 0.89 + lower_bound * 0.46 + lorawan_data_rate * 0.555)
            reliability = reliability * (1 + separation_factor - interaction_overhead) / 12
            final_reliability = reliability * 1e6

        elif self.architecture == 'Client-Server':
            network_protocol_factor = 0.15
            data_exchange_factor = 0.3
            energy = (power_setting_mote * 0.73 + sampling_rate_mote * 0.64 + spreading_factor_mote * 0.47 +
                      movement_speed_mote * 0.35 + transmission_power_threshold_mote * 0.1 + power_setting_gateway * 0.9 +
                      spreading_factor_gateway * 0.6 + transmission_power_threshold_gateway * 0.3 + shadow_fading * 0.8 +
                      path_loss_exponent * 0.65 + reference_distance * 0.33 + bandwidth_transmission * 0.58 +
                      spreading_factor_transmission * 0.69 + transmission_power_transmission * 0.91 + coding_rate * 0.15 +
                      payload_size * 0.025 + payload_length * 0.025 + signal_strength * 0.6 + shortest_distance * 0.4 +
                      number_of_rounds * 0.35 + activity_probability * 0.65 + upper_bound * 0.7 + lower_bound * 0.55 + lorawan_data_rate * 0.65) * 11
            energy = energy * (1 + data_exchange_factor - network_protocol_factor)
            reliability = 1 / (1 + power_setting_mote * 0.77 + sampling_rate_mote * 0.45 + spreading_factor_mote * 0.68 +
                               movement_speed_mote * 0.25 + transmission_power_threshold_mote * 0.60 +
                               power_setting_gateway * 0.85 + spreading_factor_gateway * 0.8 +
                               transmission_power_threshold_gateway * 0.3 + shadow_fading * 0.87 + path_loss_exponent * 0.73 +
                               reference_distance * 0.36 + bandwidth_transmission * 0.55 + spreading_factor_transmission * 0.65 +
                               transmission_power_transmission * 0.7 + coding_rate * 0.35 + payload_size * 0.05 +
                               payload_length * 0.05 + signal_strength * 0.7 + shortest_distance * 0.3 +
                               number_of_rounds * 0.75 + activity_probability * 0.55 + upper_bound * 0.89 + lower_bound * 0.46 + lorawan_data_rate * 0.555)
            reliability = reliability * (1 + network_protocol_factor - data_exchange_factor) / 11
            final_reliability = reliability * 1e6

        return energy, final_reliability

def sensitivity_analysis(problem, original_parameters, architectures):
    '''
    for architecture in architectures:
        problem.architecture = architecture
        sensitivities_energy = []
        sensitivities_reliability = []
        parameter_index = range(len(original_parameters))

        for idx in parameter_index:
            parameter_values = np.linspace(problem.xl[idx], problem.xu[idx], 10)
            energy_values = []
            reliability_values = []

            for value in parameter_values:
                params = original_parameters.copy()
                params[idx] = value

                # Handle categorical parameters properly
                if idx == 10:  # Reference distance
                    params[idx] = int(value)
                elif idx == 11:  # Bandwidth transmission
                    params[idx] = int(value)
                elif idx == 14:  # Coding rate
                    params[idx] = int(value)
                elif idx == 19:  # Number of rounds
                    params[idx] = int(value)

                energy, reliability = problem.simulate(*params)
                energy_values.append(energy)
                reliability_values.append(reliability)

            sensitivities_energy.append(energy_values)
            sensitivities_reliability.append(reliability_values)

            plt.figure(figsize=(10, 6))
            plt.plot(parameter_values, energy_values, label=f'{architecture} Energy', marker='o')
            plt.plot(parameter_values, reliability_values, label=f'{architecture} Reliability', marker='x')
            plt.xlabel(f'Parameter {idx + 1} Value')
            plt.ylabel('Objective Value')
            plt.title(f'Sensitivity Analysis for Parameter {idx + 1} in {architecture} Architecture')
            plt.legend()
            plt.show()
    '''

def sensitivity_analysis(problem, original_parameters, architectures):
    for architecture in architectures:
            problem.architecture = architecture
            sensitivities_energy = []
            sensitivities_reliability = []
            parameter_index = range(len(original_parameters))

            for idx in parameter_index:
                parameter_values = np.linspace(problem.xl[idx], problem.xu[idx], 10)
                energy_values = []
                reliability_values = []
                sensitivity_energy = []
                sensitivity_reliability = []

                original_energy, original_reliability = problem.simulate(*original_parameters)
                for value in parameter_values:
                    params = original_parameters.copy()
                    params[idx] = value

                    # Handle categorical parameters properly
                    if idx == 10:  # Reference distance
                        params[idx] = int(value)
                    elif idx == 11:  # Bandwidth transmission
                        params[idx] = int(value)
                    elif idx == 14:  # Coding rate
                        params[idx] = int(value)
                    elif idx == 19:  # Number of rounds
                        params[idx] = int(value)

                    energy, reliability = problem.simulate(*params)

                    energy_values.append(energy)
                    reliability_values.append(reliability)

                    # Sensitivity calculations - percent change in output for percent change in parameter
                    sensitivity_energy.append((energy - original_energy) / original_energy / (
                            (value - original_parameters[idx]) / original_parameters[idx]))
                    sensitivity_reliability.append((reliability - original_reliability) / original_reliability / (
                            (value - original_parameters[idx]) / original_parameters[idx]))

                sensitivities_energy.append(sum(sensitivity_energy) / len(sensitivity_energy))
                sensitivities_reliability.append(sum(sensitivity_reliability) / len(sensitivity_reliability))

                # Display parameters sorted by sensitivity
                print("Parameters sorted by sensitivity for architecture", architecture)
                print("For Energy Efficiency:")
                print(sorted(zip(parameter_index, sensitivities_energy), key=lambda x: abs(x[1]), reverse=True))
                print("For Reliability:")
                print(sorted(zip(parameter_index, sensitivities_reliability), key=lambda x: abs(x[1]), reverse=True))

                sleep(3)

def main():
        architectures = ['Component-Based', 'Layered', 'Client-Server']
        results = {}

        for architecture in architectures:
            time.sleep(10)
            problem = DingNetProblem(architecture)
            algorithm = NSGA2(pop_size=100)
            res = minimize(problem=problem, algorithm=algorithm, termination=('n_gen', 500), seed=1, save_history=True,
                           verbose=True)
            results[architecture] = res

        # Plot Pareto Front for each architecture
        colors = {'Component-Based': 'blue', 'Layered': 'green', 'Client-Server': 'red'}

        plt.figure(figsize=(10, 6))

        best_solutions = []

        for architecture in architectures:
            if results[architecture] is None or results[architecture].F is None:
                print(f"No result for {architecture}")
                continue
            F = results[architecture].F
            plt.scatter(F[:, 0], F[:, 1], c=colors[architecture], label=architecture)

            # Compute Euclidean distance to (min energy, max reliability) point, then find index of the smallest distance
            distances = np.sqrt((F[:, 0] - F[:, 0].min()) ** 2 + (F[:, 1].max() - F[:, 1]) ** 2)
            best_solution_index = np.argmin(distances)

            # Store best solutions
            best_solutions.append((architecture, F[best_solution_index, 0], F[best_solution_index, 1]))

            # Print the best solution
            print(
                f'Best solution for architecture {architecture} has energy consumption {F[best_solution_index, 0]} and reliability {F[best_solution_index, 1]}.')

            # Highlight the best solution in the plot
            plt.scatter(F[best_solution_index, 0], F[best_solution_index, 1], c='yellow', marker='x')

        if not best_solutions:
            print("No best solutions found for any architecture. Exiting...")
            return

        # Compute the most suitable architecture
        # Compute the most suitable architecture
        ideal_point = [0, 1]  # Ideal point (min Energy, max reliability)

        min_distance = np.inf
        most_suitable_architecture = None
        for architecture, energy, reliability in best_solutions:
            distance = np.sqrt((ideal_point[0] - energy) ** 2 + (ideal_point[1] - reliability) ** 2)
            if distance < min_distance:
                min_distance = distance
                most_suitable_architecture = (architecture, energy, reliability)

        print(
            f"The most suitable architecture is {most_suitable_architecture[0]} with energy consumption {most_suitable_architecture[1]} and reliability {most_suitable_architecture[2]}")

        plt.xlabel("Energy Consumption")
        plt.ylabel("Reliability")
        plt.title("Pareto Front for Different Architectures")
        plt.legend()
        plt.show()

        # Perform sensitivity analysis
        original_parameters = np.mean(np.array([ind for ind in results[architecture].X]), axis=0)
        sensitivity_analysis(problem, original_parameters, architectures)

if __name__ == "__main__":
        main()
