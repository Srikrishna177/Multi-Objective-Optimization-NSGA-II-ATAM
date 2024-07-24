from deap import creator, base, tools, algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
import matplotlib.pyplot as plt
import numpy as np
import random

class RiskAssessmentRIMOO:
    def __init__(self, pop_size, generations):
        self.pop_size = pop_size
        self.generations = generations
        self.toolbox = self.setup_toolbox()

    def calculate_client_server_risk(self):
        risk_dict = {
            "mqtt_failure": {"probability": 0.5, "impact_rel": 3, "impact_ee": 2},
            "lora_failure": {"probability": 0.5, "impact_rel": 3, "impact_ee": 3},
            "protocol_config": {"probability": 0.5, "impact_rel": 2, "impact_ee": 2},
            "congestion": {"probability": 0.6, "impact_rel": 3, "impact_ee": 2},
            "packet_loss": {"probability": 0.5, "impact_rel": 3, "impact_ee": 2},
            "capacity": {"probability": 0.4, "impact_rel": 2, "impact_ee": 1},
        }

        sum_impact_rel = sum([v["impact_rel"] for v in risk_dict.values()])
        sum_impact_ee = sum([v["impact_ee"] for v in risk_dict.values()])

        risk_rel = sum([v["probability"] * v["impact_rel"] for v in risk_dict.values()]) / sum_impact_rel
        risk_ee = sum([v["probability"] * v["impact_ee"] for v in risk_dict.values()]) / sum_impact_ee

        return risk_rel, risk_ee

    def calculate_component_based_architecture_risk(self):
        risk_dict = {
            "incorrect_power_setting_mote": {"probability": 0.4, "impact_rel": 6, "impact_ee": 8},
            "incorrect_spreading_factor_mote": {"probability": 0.5, "impact_rel": 7, "impact_ee": 6},
            "incorrect_sampling_rate_mote": {"probability": 0.3, "impact_rel": 6, "impact_ee": 6},
            "incorrect_movement_speed_mote": {"probability": 0.4, "impact_rel": 5, "impact_ee": 3},
            "incorrect_transmission_power_threshold_mote": {"probability": 0.3, "impact_rel": 7,
                                                            "impact_ee": 8},
            "unauthorized_access_mote": {"probability": 0.5, "impact_rel": 7, "impact_ee": 6},
            "incorrect_power_setting_gateway": {"probability": 0.9, "impact_rel": 6, "impact_ee": 8},
            "incorrect_spreading_factor_gateway": {"probability": 0.8, "impact_rel": 7, "impact_ee": 6},
            "wrong_values_sampling_rate_mote": {"probability": 0.8, "impact_rel": 6, "impact_ee": 6},
            "high_movement_speed_mote": {"probability": 0.9, "impact_rel": 5, "impact_ee": 3},
            "wrong_values_transmission_power_threshold_gateway": {"probability": 0.4, "impact_rel": 7,
                                                                  "impact_ee": 8},
            "network_interference": {"probability": 0.9, "impact_rel": 6, "impact_ee": 8},
            "unauthorized_access_gateway": {"probability": 0.8, "impact_rel": 7, "impact_ee": 6},
            "Misconfiguration Causing Unreliable Communication": {"probability": 0.8, "impact_rel": 7,
                                                                  "impact_ee": 8},
            "improper_spreading_factor_transmission": {"probability": 0.5, "impact_rel": 7,
                                                       "impact_ee": 8},
            "incorrect_bandwidth_transmission_settings": {"probability": 0.4, "impact_rel": 6, "impact_ee": 8},
            "incorrect_transmission_power_transmission_settings": {"probability": 0.5, "impact_rel": 7, "impact_ee": 6},
            "incorrect_coding_rate_transmission": {"probability": 0.3, "impact_rel": 5, "impact_ee": 3},
            "payload_length": {"probability": 0.4, "impact_rel": 7, "impact_ee": 6},
            "payload_size": {"probability": 0.4, "impact_rel": 6, "impact_ee": 6},
            "server_overload": {"probability": 0.5, "impact_rel": 8, "impact_ee": 6},
            "delayed_message_delivery": {"probability": 0.6, "impact_rel": 7, "impact_ee": 6},
            "unauthorized_access": {"probability": 0.5, "impact_rel": 9, "impact_ee": 6},
            "message_loss": {"probability": 0.6, "impact_rel": 8, "impact_ee": 6},
            "incorrect_message_format": {"probability": 0.4, "impact_rel": 6, "impact_ee": 6},
            "inaccurate_signal_strength": {"probability": 0.6, "impact_rel": 8, "impact_ee": 7},
            "incorrect_distance_measurement": {"probability": 0.5, "impact_rel": 7, "impact_ee": 6},
            "signal_fading": {"probability": 0.5, "impact_rel": 9, "impact_ee": 8},
            "dynamic_obstacles": {"probability": 0.6, "impact_rel": 8, "impact_ee": 7},
            "incorrect_number_of_rounds": {"probability": 0.3, "impact_rel": 6, "impact_ee": 3},
            "incorrect_number_of_motes": {"probability": 0.4, "impact_rel": 7, "impact_ee": 5},
            "incorrect_activity_probability": {"probability": 0.3, "impact_rel": 5, "impact_ee": 4},
            "incorrect_upper_bound_setting": {"probability": 0.5, "impact_rel": 2, "impact_ee": 4},
            "excessive_upper_bound": {"probability": 0.4, "impact_rel": 7, "impact_ee": 7},
            "inadequate_upper_bound": {"probability": 0.4, "impact_rel": 8, "impact_ee": 5},
            "incorrect_lower_bound_setting": {"probability": 0.5, "impact_rel": 6, "impact_ee": 8},
            "excessive_lower_bound": {"probability": 0.4, "impact_rel": 9, "impact_ee": 7},
            "inadequate_lower_bound": {"probability": 0.4, "impact_rel": 5, "impact_ee": 6},
        }

        sum_impact_rel_cba = sum([v["impact_rel"] for v in risk_dict.values()])
        sum_impact_ee_cba = sum([v["impact_ee"] for v in risk_dict.values()])

        risk_rel_cba = sum([v["probability"] * v["impact_rel"] for v in risk_dict.values()]) / sum_impact_rel_cba
        risk_ee_cba = sum([v["probability"] * v["impact_ee"] for v in risk_dict.values()]) / sum_impact_ee_cba

        return risk_rel_cba, risk_ee_cba

    def calculate_layered_risk(self):
        risk_dict = {
            "incorrect_configuration values": {"probability": 0.6, "impact_rel": 7, "impact_ee": 3},
            "invalid_input_profiles": {"probability": 0.5, "impact_rel": 5, "impact_ee": 5},
            "simulation_data_errors": {"probability": 0.5, "impact_rel": 7, "impact_ee": 3},
            "incorrect_adaptation_strategy": {"probability": 0.7, "impact_rel": 6, "impact_ee": 6},
            "incorrect_input_profiles": {"probability": 0.7, "impact_rel": 6, "impact_ee": 3},
            "incorrect_upper_bound": {"probability": 0.7, "impact_rel": 4, "impact_ee": 6},
            "incorrect_lower_bound": {"probability": 0.7, "impact_rel": 4, "impact_ee": 6},
            "simulation_errors": {"probability": 0.7, "impact_rel": 7, "impact_ee": 3},
            "packet_loss_due_to_inference": {"probability": 0.7, "impact_rel": 7, "impact_ee": 5},
            "high_latency": {"probability": 0.3, "impact_rel": 5, "impact_ee": 3},
            "message_queue_workflow": {"probability": 0.5, "impact_rel": 7, "impact_ee": 3},
            "broker_failure": {"probability": 0.3, "impact_rel": 7, "impact_ee": 3},
            "incorrect_map_rendering_env_controller": {"probability": 0.8, "impact_rel": 7, "impact_ee": 3},
            "incorrect_power_setting_mote_controller": {"probability": 0.9, "impact_rel": 6, "impact_ee": 8},
            "incorrect_spreading_factor_mote_controller": {"probability": 0.8, "impact_rel": 7, "impact_ee": 6},
            "incorrect_sampling_rate_mote_controller": {"probability": 0.8, "impact_rel": 6, "impact_ee": 6},
            "incorrect_movement_speed_mote_controller": {"probability": 0.6, "impact_rel": 5, "impact_ee": 3},
            "incorrect_transmission_power_threshold_mote_controller": {"probability": 0.2, "impact_rel": 7,
                                                                       "impact_ee": 8},
            "incorrect_power_setting_gateway_controller": {"probability": 0.9, "impact_rel": 6, "impact_ee": 8},
            "incorrect_spreading_factor_gateway_controller": {"probability": 0.8, "impact_rel": 7, "impact_ee": 6},
            "incorrect_transmission_power_threshold_gateway_controller": {"probability": 0.8, "impact_rel": 7,
                                                                          "impact_ee": 8},
            "incorrect_shadow_fading_controller": {"probability": 0.5, "impact_rel": 7, "impact_ee": 3},
            "incorrect_path_loss_controller": {"probability": 0.6, "impact_rel": 7, "impact_ee": 3},
            "incorrect_reference_distance_controller": {"probability": 0.7, "impact_rel": 7, "impact_ee": 3},
            "incorrect_adaptation_strategy_controller": {"probability": 0.8, "impact_rel": 6, "impact_ee": 6},
            "incorrect_input_profiles_controller": {"probability": 0.7, "impact_rel": 5, "impact_ee": 2},
            "incorrect_upper_bound_controller": {"probability": 0.7, "impact_rel": 2, "impact_ee": 4},
            "incorrect_lower_bound_controller": {"probability": 0.65, "impact_rel": 3, "impact_ee": 5},
            "incorrect_simulation_logic": {"probability": 0.8, "impact_rel": 8, "impact_ee": 3},
            "incorrect_map_rendering_env_view": {"probability": 0.4, "impact_rel": 7, "impact_ee": 3},
            "incorrect_power_setting_mote_view": {"probability": 0.6, "impact_rel": 6, "impact_ee": 8},
            "incorrect_spreading_factor_mote_view": {"probability": 0.5, "impact_rel": 7, "impact_ee": 6},
            "incorrect_sampling_rate_mote_view": {"probability": 0.5, "impact_rel": 6, "impact_ee": 6},
            "incorrect_movement_speed_mote_view": {"probability": 0.3, "impact_rel": 5, "impact_ee": 3},
            "incorrect_transmission_power_threshold_mote_view": {"probability": 0.1, "impact_rel": 7,
                                                                 "impact_ee": 8},
            "incorrect_power_setting_gateway_view": {"probability": 0.6, "impact_rel": 6, "impact_ee": 8},
            "incorrect_spreading_factor_gateway_view": {"probability": 0.5, "impact_rel": 7, "impact_ee": 6},
            "incorrect_transmission_power_threshold_gateway_view": {"probability": 0.2, "impact_rel": 7,
                                                                    "impact_ee": 8},
            "incorrect_shadow_fading_view": {"probability": 0.4, "impact_rel": 7, "impact_ee": 3},
            "incorrect_path_loss_view": {"probability": 0.5, "impact_rel": 7, "impact_ee": 3},
            "incorrect_reference_distance_view": {"probability": 0.4, "impact_rel": 7, "impact_ee": 3},
            "incorrect_adaptation_strategy_view": {"probability": 0.5, "impact_rel": 6, "impact_ee": 6},
            "incorrect_input_profiles_view": {"probability": 0.5, "impact_rel": 5, "impact_ee": 2},
            "incorrect_upper_bound_view": {"probability": 0.5, "impact_rel": 2, "impact_ee": 4},
            "incorrect_lower_bound_view": {"probability": 0.55, "impact_rel": 3, "impact_ee": 5},
            "incorrect_simulation_logic_view": {"probability": 0.5, "impact_rel": 8, "impact_ee": 3},
        }

        sum_impact_rel_la = sum([v["impact_rel"] for v in risk_dict.values()])
        sum_impact_ee_la = sum([v["impact_ee"] for v in risk_dict.values()])

        risk_rel_la = sum([v["probability"] * v["impact_rel"] for v in risk_dict.values()]) / sum_impact_rel_la
        risk_ee_la = sum([v["probability"] * v["impact_ee"] for v in risk_dict.values()]) / sum_impact_ee_la

        return risk_rel_la, risk_ee_la

    def calculate_risks(self, architecture):
        if architecture == "client_server":
            return self.calculate_client_server_risk()
        elif architecture == "component_based":
            return self.calculate_component_based_architecture_risk()
        elif architecture == "layered":
            return self.calculate_layered_risk()



    def evaluate(self, individual):
        architecture = individual[0]
        risk_reliability, risk_energy_efficiency = self.calculate_risks(architecture)
        return risk_reliability, risk_energy_efficiency

    def setup_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_architecture", random.choice, ["client_server", "component_based", "layered"])
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_architecture, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        return toolbox

    def plot_pareto_front(self, population):
        fitnesses = [ind.fitness.values for ind in population]
        risk_rel, risk_ee = zip(*fitnesses)

        plt.scatter(risk_rel, risk_ee, color='b')
        plt.axis("tight")
        plt.xlabel("Reliability Risk")
        plt.ylabel("Energy Efficiency Risk")
        plt.title("Pareto Front")
        plt.show()

    def optimize(self):
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.ParetoFront()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min, axis=0)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=self.generations,
                                       stats=stats, halloffame=hof, verbose=True)

        return pop, log, hof


if __name__ == "__main__":
    risk_assessment = RiskAssessmentRIMOO(100, 50)
    pop, log, hof = risk_assessment.optimize()
    risk_assessment.plot_pareto_front(hof)