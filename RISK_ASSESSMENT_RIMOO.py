from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import numpy as np
import time
from SALib.analyze import sobol
from SALib.sample import sobol
from pymoo.config import Config
Config.warnings['not_compiled'] = False
from joblib import Parallel, delayed

class RiskAssessmentRIMOO(Problem):  # Insert your actual superclass name if RiskAssessmentRIMOO is a subclass.
    def __init__(self, n_var, n_obj, n_constr, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

class ComponentBasedRIMOO(RiskAssessmentRIMOO):
    def __init__(self):
        super().__init__(33, 2, 0, np.array([1] * 33), np.array([1000] * 33))

    def _evaluate(self, X, out, *args, **kwargs):
        energy_consumption_value = []
        reliability_value = []

        for x in X:
            (Incorrect_power_settings_mote_CBA, Inappropriate_Spreading_Factor_Mote_CBA, Incorrect_Sampling_Rate_Mote_CBA, High_Movement_Speed_Mote_CBA, Incorrect_Transmission_Power_Threshold_Mote_CBA, Unauthorized_Access_CBA, Incorrect_Power_Setting_Gateway_CBA, Inappropriate_Spreading_Factor_Gateway_CBA, Incorrect_Transmission_Power_Threshold_Gateway_CBA, Incorrect_Shadow_Fading_Values_CBA, Inaccurate_Path_Loss_Exponent_CBA, Incorrect_Reference_Distance_CBA, Misconfiguration_Color_Settings_CBA, Interaction_Between_Shadow_Fading_Path_Loss_Exponent_Reference_Distance_CBA, Incorrect_Bandwidth_Settings_LoraWan_CBA, Improper_Spreading_Factor_LoraWan_CBA, Incorrect_Transmission_Power_Settings_LoraWan_CBA, Incorrect_coding_rate_CBA, Excessive_payload_size_CBA, Incorrect_payload_length_CBA, Buffer_Overflow_CBA, Subscription_Handling_Error_CBA, Delayed_Message_Processing_CBA, Security_Breach_CBA, Inaccurate_Signal_Strength_CBA, Incorrect_Distance_Measurement_CBA, Signal_Fading_CBA, Dynamic_Obstacles_CBA, Incorrect_Number_of_Rounds_CBA, Incorrect_Number_of_Motes_CBA, Incorrect_Activity_Probability_CBA, Incorrect_Upper_Bound_Value_CBA, Incorrect_Lower_Bound_Value_CBA) = x
            energy, reliability = self.simulate(Incorrect_power_settings_mote_CBA,
                                                Inappropriate_Spreading_Factor_Mote_CBA,
                                                Incorrect_Sampling_Rate_Mote_CBA, High_Movement_Speed_Mote_CBA,
                                                Incorrect_Transmission_Power_Threshold_Mote_CBA,
                                                Unauthorized_Access_CBA, Incorrect_Power_Setting_Gateway_CBA,
                                                Inappropriate_Spreading_Factor_Gateway_CBA,
                                                Incorrect_Transmission_Power_Threshold_Gateway_CBA,
                                                Incorrect_Shadow_Fading_Values_CBA, Inaccurate_Path_Loss_Exponent_CBA,
                                                Incorrect_Reference_Distance_CBA, Misconfiguration_Color_Settings_CBA,
                                                Interaction_Between_Shadow_Fading_Path_Loss_Exponent_Reference_Distance_CBA,
                                                Incorrect_Bandwidth_Settings_LoraWan_CBA,
                                                Improper_Spreading_Factor_LoraWan_CBA,
                                                Incorrect_Transmission_Power_Settings_LoraWan_CBA,
                                                Incorrect_coding_rate_CBA, Excessive_payload_size_CBA,
                                                Incorrect_payload_length_CBA, Buffer_Overflow_CBA,
                                                Subscription_Handling_Error_CBA, Delayed_Message_Processing_CBA,
                                                Security_Breach_CBA, Inaccurate_Signal_Strength_CBA,
                                                Incorrect_Distance_Measurement_CBA, Signal_Fading_CBA,
                                                Dynamic_Obstacles_CBA, Incorrect_Number_of_Rounds_CBA,
                                                Incorrect_Number_of_Motes_CBA, Incorrect_Activity_Probability_CBA,
                                                Incorrect_Upper_Bound_Value_CBA, Incorrect_Lower_Bound_Value_CBA)

            energy_consumption_value.append(energy)
            reliability_value.append(reliability)

        out["F"] = np.column_stack([energy_consumption_value, reliability_value])

    def simulate(self, Incorrect_power_settings_mote_CBA, Inappropriate_Spreading_Factor_Mote_CBA, Incorrect_Sampling_Rate_Mote_CBA, High_Movement_Speed_Mote_CBA,
                                                Incorrect_Transmission_Power_Threshold_Mote_CBA,
                                                Unauthorized_Access_CBA, Incorrect_Power_Setting_Gateway_CBA,
                                                Inappropriate_Spreading_Factor_Gateway_CBA,
                                                Incorrect_Transmission_Power_Threshold_Gateway_CBA,
                                                Incorrect_Shadow_Fading_Values_CBA, Inaccurate_Path_Loss_Exponent_CBA,
                                                Incorrect_Reference_Distance_CBA, Misconfiguration_Color_Settings_CBA,
                                                Interaction_Between_Shadow_Fading_Path_Loss_Exponent_Reference_Distance_CBA,
                                                Incorrect_Bandwidth_Settings_LoraWan_CBA,
                                                Improper_Spreading_Factor_LoraWan_CBA,
                                                Incorrect_Transmission_Power_Settings_LoraWan_CBA,
                                                Incorrect_coding_rate_CBA, Excessive_payload_size_CBA,
                                                Incorrect_payload_length_CBA, Buffer_Overflow_CBA,
                                                Subscription_Handling_Error_CBA, Delayed_Message_Processing_CBA,
                                                Security_Breach_CBA, Inaccurate_Signal_Strength_CBA,
                                                Incorrect_Distance_Measurement_CBA, Signal_Fading_CBA,
                                                Dynamic_Obstacles_CBA, Incorrect_Number_of_Rounds_CBA,
                                                Incorrect_Number_of_Motes_CBA, Incorrect_Activity_Probability_CBA,
                                                Incorrect_Upper_Bound_Value_CBA, Incorrect_Lower_Bound_Value_CBA):
        reusability_factor = 0.15
        loosely_coupling_factor = 0.05
        energy = (0.305 * Incorrect_power_settings_mote_CBA + 0.4 * Inappropriate_Spreading_Factor_Mote_CBA + 0.155 * Incorrect_Sampling_Rate_Mote_CBA + 0.28 * High_Movement_Speed_Mote_CBA + 0.175 * Incorrect_Transmission_Power_Threshold_Mote_CBA + 0.4 * Unauthorized_Access_CBA + 0.245 * Incorrect_Power_Setting_Gateway_CBA + 0.26 * Inappropriate_Spreading_Factor_Gateway_CBA + 0.14 * Incorrect_Transmission_Power_Threshold_Gateway_CBA + 0.51 * Incorrect_Shadow_Fading_Values_CBA + 0.4 * Inaccurate_Path_Loss_Exponent_CBA + 0.18 * Incorrect_Reference_Distance_CBA + 0.29 * Misconfiguration_Color_Settings_CBA + 0.23 * Interaction_Between_Shadow_Fading_Path_Loss_Exponent_Reference_Distance_CBA + 0.21 * Incorrect_Bandwidth_Settings_LoraWan_CBA + 0.40 * Improper_Spreading_Factor_LoraWan_CBA + 0.40 * Incorrect_Transmission_Power_Settings_LoraWan_CBA + 0.16 * Incorrect_coding_rate_CBA + 0.24 * Excessive_payload_size_CBA + 0.233 * Incorrect_payload_length_CBA + 0.77 * Buffer_Overflow_CBA + 0.147 * Subscription_Handling_Error_CBA + 0.60 * Delayed_Message_Processing_CBA + 0.8 * Security_Breach_CBA + 0.46 * Inaccurate_Signal_Strength_CBA + 0.40 * Incorrect_Distance_Measurement_CBA + 0.40 * Signal_Fading_CBA + 0.43 * Dynamic_Obstacles_CBA + 0.185 * Incorrect_Number_of_Rounds_CBA + 0.25 * Incorrect_Number_of_Motes_CBA + 0.155 * Incorrect_Activity_Probability_CBA + 0.31 * Incorrect_Upper_Bound_Value_CBA + 0.17 * Incorrect_Lower_Bound_Value_CBA) * 10
        energy = energy * (1 - reusability_factor + loosely_coupling_factor)
        reliability = 1 / (
                    1 + 0.305 * Incorrect_power_settings_mote_CBA + 0.4 * Inappropriate_Spreading_Factor_Mote_CBA + 0.155 * Incorrect_Sampling_Rate_Mote_CBA + 0.28 * High_Movement_Speed_Mote_CBA + 0.175 * Incorrect_Transmission_Power_Threshold_Mote_CBA + 0.4 * Unauthorized_Access_CBA + 0.245 * Incorrect_Power_Setting_Gateway_CBA + 0.26 * Inappropriate_Spreading_Factor_Gateway_CBA + 0.14 * Incorrect_Transmission_Power_Threshold_Gateway_CBA + 0.51 * Incorrect_Shadow_Fading_Values_CBA + 0.4 * Inaccurate_Path_Loss_Exponent_CBA + 0.18 * Incorrect_Reference_Distance_CBA + 0.29 * Misconfiguration_Color_Settings_CBA + 0.23 * Interaction_Between_Shadow_Fading_Path_Loss_Exponent_Reference_Distance_CBA + 0.21 * Incorrect_Bandwidth_Settings_LoraWan_CBA + 0.40 * Improper_Spreading_Factor_LoraWan_CBA + 0.40 * Incorrect_Transmission_Power_Settings_LoraWan_CBA + 0.16 * Incorrect_coding_rate_CBA + 0.24 * Excessive_payload_size_CBA + 0.233 * Incorrect_payload_length_CBA + 0.77 * Buffer_Overflow_CBA + 0.147 * Subscription_Handling_Error_CBA + 0.60 * Delayed_Message_Processing_CBA + 0.8 * Security_Breach_CBA + 0.46 * Inaccurate_Signal_Strength_CBA + 0.40 * Incorrect_Distance_Measurement_CBA + 0.40 * Signal_Fading_CBA + 0.43 * Dynamic_Obstacles_CBA + 0.185 * Incorrect_Number_of_Rounds_CBA + 0.25 * Incorrect_Number_of_Motes_CBA + 0.155 * Incorrect_Activity_Probability_CBA + 0.31 * Incorrect_Upper_Bound_Value_CBA + 0.17 * Incorrect_Lower_Bound_Value_CBA) / 10
        reliability = reliability * (1 + reusability_factor - loosely_coupling_factor)
        final_reliability = reliability * 1e6

        return energy, final_reliability
    pass

class LayeredRIMOO(RiskAssessmentRIMOO):
    def __init__(self):
        super().__init__(40, 2, 0, np.array([1] * 40), np.array([18] * 40))

    def _evaluate(self, X, out, *args, **kwargs):
        energy_consumption_value = []
        reliability_value = []

        for x in X:
            (Incorrect_region_mapping_LA, Incorrect_power_settings_mote_LA, Latency_in_Data_Update_LA, Inappropriate_Spreading_Factor_Mote_LA, Improper_Sampling_Rate_Mote_LA, High_Movement_Speed_Mote_LA, Incorrect_Power_Threshold_Mote_LA, Incorrect_Power_Setting_Gateway_LA, Inappropriate_Spreading_Factor_Gateway_LA, Incorrect_Power_Threshold_Gateway_LA, Incorrect_Shadow_Fading_Setting_LA, Incorrect_Path_Loss_Exponent_LA, Incorrect_Reference_Distance_LA, Incorrect_Signal_Strength_Threshold_LA, Incorrect_Distance_Measurement_LA, Incorrect_Activity_Probability_LA, Incorrect_Upper_Bound_Setting_LA, Incorrect_Lower_Bound_Setting_LA, Incorrect_Simulation_Parameters_LA, Incorrect_Cumulative_Results_LA, Incorrect_Bandwidth_Settings_of_LoraWan_LA, Improper_Spreading_Factor_of_LoraWan_LA, Incorrect_Transmission_Power_of_LoraWan_LA, Incorrect_Coding_Rate_of_LoraWan_LA, Excessive_Payload_Size_of_LoraWan_LA, Incorrect_Payload_Length_of_LoraWan_LA, Server_Downtime_LA, Incorrect_Quality_of_Service_Settings_LA, Network_Congestion_LA, Security_Breaches_LA, Incorrect_adaptation_strategy_logic_LA, Incorrect_Input_Profiles_LA, Misconfigured_adaptation_goals_intervals_LA, Simulation_Logic_Errors_LA, Incorrect_Configuration_Settings_LA, File_Corruption_Or_Unreadable_File_Format_LA, Inaccurate_Input_Profiles_Data_LA, Missing_Or_Incomplete_Input_Profiles_LA, Incorrect_Simulation_Parameters_LA, Simulation_Run_Data_Corruption_LA) = x
            energy, reliability = self.simulate(Incorrect_region_mapping_LA, Incorrect_power_settings_mote_LA, Latency_in_Data_Update_LA, Inappropriate_Spreading_Factor_Mote_LA, Improper_Sampling_Rate_Mote_LA, High_Movement_Speed_Mote_LA, Incorrect_Power_Threshold_Mote_LA, Incorrect_Power_Setting_Gateway_LA, Inappropriate_Spreading_Factor_Gateway_LA, Incorrect_Power_Threshold_Gateway_LA, Incorrect_Shadow_Fading_Setting_LA, Incorrect_Path_Loss_Exponent_LA, Incorrect_Reference_Distance_LA, Incorrect_Signal_Strength_Threshold_LA, Incorrect_Distance_Measurement_LA, Incorrect_Activity_Probability_LA, Incorrect_Upper_Bound_Setting_LA, Incorrect_Lower_Bound_Setting_LA, Incorrect_Simulation_Parameters_LA, Incorrect_Cumulative_Results_LA, Incorrect_Bandwidth_Settings_of_LoraWan_LA, Improper_Spreading_Factor_of_LoraWan_LA, Incorrect_Transmission_Power_of_LoraWan_LA, Incorrect_Coding_Rate_of_LoraWan_LA, Excessive_Payload_Size_of_LoraWan_LA, Incorrect_Payload_Length_of_LoraWan_LA, Server_Downtime_LA, Incorrect_Quality_of_Service_Settings_LA, Network_Congestion_LA, Security_Breaches_LA, Incorrect_adaptation_strategy_logic_LA, Incorrect_Input_Profiles_LA, Misconfigured_adaptation_goals_intervals_LA, Simulation_Logic_Errors_LA, Incorrect_Configuration_Settings_LA, File_Corruption_Or_Unreadable_File_Format_LA, Inaccurate_Input_Profiles_Data_LA, Missing_Or_Incomplete_Input_Profiles_LA, Incorrect_Simulation_Parameters_LA, Simulation_Run_Data_Corruption_LA)

            energy_consumption_value.append(energy)
            reliability_value.append(reliability)

        out["F"] = np.column_stack([energy_consumption_value, reliability_value])

    def simulate(self, Incorrect_region_mapping_LA, Incorrect_power_settings_mote_LA, Latency_in_Data_Update_LA, Inappropriate_Spreading_Factor_Mote_LA, Improper_Sampling_Rate_Mote_LA, High_Movement_Speed_Mote_LA, Incorrect_Power_Threshold_Mote_LA, Incorrect_Power_Setting_Gateway_LA, Inappropriate_Spreading_Factor_Gateway_LA, Incorrect_Power_Threshold_Gateway_LA, Incorrect_Shadow_Fading_Setting_LA, Incorrect_Path_Loss_Exponent_LA, Incorrect_Reference_Distance_LA, Incorrect_Signal_Strength_Threshold_LA, Incorrect_Distance_Measurement_LA, Incorrect_Activity_Probability_LA, Inappropriate_number_of_rounds_LA, Incorrect_Upper_Bound_Setting_LA, Incorrect_Lower_Bound_Setting_LA, Incorrect_Simulation_Parameters_LA, Incorrect_Cumulative_Results_LA, Incorrect_Bandwidth_Settings_of_LoraWan_LA, Improper_Spreading_Factor_of_LoraWan_LA, Incorrect_Transmission_Power_of_LoraWan_LA, Incorrect_Coding_Rate_of_LoraWan_LA, Excessive_Payload_Size_of_LoraWan_LA, Incorrect_Payload_Length_of_LoraWan_LA, Server_Downtime_LA, Incorrect_Quality_of_Service_Settings_LA, Network_Congestion_LA, Security_Breaches_LA, Incorrect_adaptation_strategy_logic_LA, Incorrect_Input_Profiles_LA, Misconfigured_adaptation_goals_intervals_LA, Simulation_Logic_Errors_LA, Incorrect_Configuration_Settings_LA, File_Corruption_Or_Unreadable_File_Format_LA, Inaccurate_Input_Profiles_Data_LA, Missing_Or_Incomplete_Input_Profiles_LA, Simulation_Run_Data_Corruption_LA):
        SoC_factor = 0.25
        interaction_overhead_factor = 0.1
        energy = (0.8 * Incorrect_region_mapping_LA + 1.0 * Incorrect_power_settings_mote_LA + 0.6 * Latency_in_Data_Update_LA + 0.8 * Inappropriate_Spreading_Factor_Mote_LA + 0.6 * Improper_Sampling_Rate_Mote_LA + 0.8 * High_Movement_Speed_Mote_LA + 0.6 * Incorrect_Power_Threshold_Mote_LA + 0.8 * Incorrect_Power_Setting_Gateway_LA + 0.8 * Inappropriate_Spreading_Factor_Gateway_LA + 0.6 * Incorrect_Power_Threshold_Gateway_LA + 0.6 * Incorrect_Shadow_Fading_Setting_LA + 0.8 * Incorrect_Path_Loss_Exponent_LA + 0.6 * Incorrect_Reference_Distance_LA + 0.8 * Incorrect_Signal_Strength_Threshold_LA + 0.8 * Incorrect_Distance_Measurement_LA + 0.7 * Incorrect_Activity_Probability_LA + 0.7 * Inappropriate_number_of_rounds_LA + 0.8 * Incorrect_Upper_Bound_Setting_LA + 0.8 * Incorrect_Lower_Bound_Setting_LA + 0.6 * Incorrect_Simulation_Parameters_LA + 0.6 * Incorrect_Cumulative_Results_LA + 0.4 * Incorrect_Bandwidth_Settings_of_LoraWan_LA + 0.3 * Improper_Spreading_Factor_of_LoraWan_LA + 0.5 * Incorrect_Transmission_Power_of_LoraWan_LA + 0.3 * Incorrect_Coding_Rate_of_LoraWan_LA + 0.2 * Excessive_Payload_Size_of_LoraWan_LA + 0.2 * Incorrect_Payload_Length_of_LoraWan_LA + 0.4 * Server_Downtime_LA + 0.3 * Incorrect_Quality_of_Service_Settings_LA + 0.8 * Network_Congestion_LA + 0.2 * Security_Breaches_LA + 0.5 * Incorrect_adaptation_strategy_logic_LA + 0.4 * Incorrect_Input_Profiles_LA + 0.45 * Misconfigured_adaptation_goals_intervals_LA + 0.6 * Simulation_Logic_Errors_LA + 0.8 * Incorrect_Configuration_Settings_LA + 0.7 * File_Corruption_Or_Unreadable_File_Format_LA + 0.8 * Inaccurate_Input_Profiles_Data_LA + 0.6 * Missing_Or_Incomplete_Input_Profiles_LA + 0.7 * Simulation_Run_Data_Corruption_LA) * 12
        energy = energy * (1 - SoC_factor + interaction_overhead_factor)
        reliability = 1 / (
                    1 + 0.8 * Incorrect_region_mapping_LA + 1.0 * Incorrect_power_settings_mote_LA + 0.6 * Latency_in_Data_Update_LA + 0.8 * Inappropriate_Spreading_Factor_Mote_LA + 0.6 * Improper_Sampling_Rate_Mote_LA + 0.8 * High_Movement_Speed_Mote_LA + 0.6 * Incorrect_Power_Threshold_Mote_LA + 0.8 * Incorrect_Power_Setting_Gateway_LA + 0.8 * Inappropriate_Spreading_Factor_Gateway_LA + 0.6 * Incorrect_Power_Threshold_Gateway_LA + 0.6 * Incorrect_Shadow_Fading_Setting_LA + 0.8 * Incorrect_Path_Loss_Exponent_LA + 0.6 * Incorrect_Reference_Distance_LA + 0.8 * Incorrect_Signal_Strength_Threshold_LA + 0.8 * Incorrect_Distance_Measurement_LA + 0.7 * Incorrect_Activity_Probability_LA + 0.7 * Inappropriate_number_of_rounds_LA + 0.8 * Incorrect_Upper_Bound_Setting_LA + 0.8 * Incorrect_Lower_Bound_Setting_LA + 0.6 * Incorrect_Simulation_Parameters_LA + 0.6 * Incorrect_Cumulative_Results_LA + 0.4 * Incorrect_Bandwidth_Settings_of_LoraWan_LA + 0.3 * Improper_Spreading_Factor_of_LoraWan_LA + 0.5 * Incorrect_Transmission_Power_of_LoraWan_LA + 0.3 * Incorrect_Coding_Rate_of_LoraWan_LA + 0.2 * Excessive_Payload_Size_of_LoraWan_LA + 0.2 * Incorrect_Payload_Length_of_LoraWan_LA + 0.4 * Server_Downtime_LA + 0.3 * Incorrect_Quality_of_Service_Settings_LA + 0.8 * Network_Congestion_LA + 0.2 * Security_Breaches_LA + 0.5 * Incorrect_adaptation_strategy_logic_LA + 0.4 * Incorrect_Input_Profiles_LA + 0.45 * Misconfigured_adaptation_goals_intervals_LA + 0.6 * Simulation_Logic_Errors_LA + 0.8 * Incorrect_Configuration_Settings_LA + 0.7 * File_Corruption_Or_Unreadable_File_Format_LA + 0.8 * Inaccurate_Input_Profiles_Data_LA + 0.6 * Missing_Or_Incomplete_Input_Profiles_LA + 0.7 * Simulation_Run_Data_Corruption_LA) / 12
        reliability = reliability * (1 + SoC_factor - interaction_overhead_factor)
        final_reliability = reliability * 1e6

        return energy, final_reliability
    pass

class ClientServerRIMOO(RiskAssessmentRIMOO):
    def __init__(self):
        super().__init__(6, 2, 0, np.array([1] * 6), np.array([18] * 6))

    def _evaluate(self, X, out, *args, **kwargs):
        energy_consumption_value = []
        reliability_value = []

        for x in X:
            (MQTT_Protocol_Failure_CSA, LoRaWaN_Protocol_Failure_CSA, Incorrect_Protocol_Configuration_CSA, High_Data_Volumes_Causing_Network_Congestion_CSA, Data_Packet_Loss_CSA, Insufficient_Data_Handling_Capacity_CSA) = x
            energy, reliability = self.simulate(MQTT_Protocol_Failure_CSA, LoRaWaN_Protocol_Failure_CSA, Incorrect_Protocol_Configuration_CSA, High_Data_Volumes_Causing_Network_Congestion_CSA, Data_Packet_Loss_CSA, Insufficient_Data_Handling_Capacity_CSA)

            energy_consumption_value.append(energy)
            reliability_value.append(reliability)

        out["F"] = np.column_stack([energy_consumption_value, reliability_value])

    def simulate(self, MQTT_Protocol_Failure_CSA, LoRaWaN_Protocol_Failure_CSA, Incorrect_Protocol_Configuration_CSA, High_Data_Volumes_Causing_Network_Congestion_CSA, Data_Packet_Loss_CSA,  Insufficient_Data_Handling_Capacity_CSA):
        network_protocol_usage_factor = 0.15
        data_exchange_volume_factor = 0.3
        energy = (
                             0.5 * MQTT_Protocol_Failure_CSA + 0.5 * LoRaWaN_Protocol_Failure_CSA + 0.5 * Incorrect_Protocol_Configuration_CSA + 0.6 * High_Data_Volumes_Causing_Network_Congestion_CSA + 0.5 * Data_Packet_Loss_CSA + 0.4 * Insufficient_Data_Handling_Capacity_CSA) * 11
        energy = energy * (1 - network_protocol_usage_factor + data_exchange_volume_factor)
        reliability = 1 / (
                    1 + 0.5 * MQTT_Protocol_Failure_CSA + 0.5 * LoRaWaN_Protocol_Failure_CSA + 0.5 * Incorrect_Protocol_Configuration_CSA + 0.6 * High_Data_Volumes_Causing_Network_Congestion_CSA + 0.5 * Data_Packet_Loss_CSA + 0.4 * Insufficient_Data_Handling_Capacity_CSA) / 11
        reliability = reliability * (1 + network_protocol_usage_factor - data_exchange_volume_factor)
        final_reliability = reliability * 1e6

        return energy, final_reliability
    pass

def run_sensitivity_analysis():
    architectures = [ComponentBasedRIMOO, LayeredRIMOO, ClientServerRIMOO]
    architecture_names = ['Component-Based', 'Layered', 'Client-Server']

    params = {
        'num_vars': [33, 40, 6],
        'names': [['Incorrect_power_settings_mote_CBA', 'Inappropriate_Spreading_Factor_Mote_CBA', 'Incorrect_Sampling_Rate_Mote_CBA', 'High_Movement_Speed_Mote_CBA',
                                                'Incorrect_Transmission_Power_Threshold_Mote_CBA',
                                                'Unauthorized_Access_CBA', 'Incorrect_Power_Setting_Gateway_CBA',
                                                'Inappropriate_Spreading_Factor_Gateway_CBA',
                                                'Incorrect_Transmission_Power_Threshold_Gateway_CBA',
                                                'Incorrect_Shadow_Fading_Values_CBA', 'Inaccurate_Path_Loss_Exponent_CBA',
                                                'Incorrect_Reference_Distance_CBA', 'Misconfiguration_Color_Settings_CBA',
                                                'Interaction_Between_Shadow_Fading_Path_Loss_Exponent_Reference_Distance_CBA',
                                                'Incorrect_Bandwidth_Settings_LoraWan_CBA',
                                                'Improper_Spreading_Factor_LoraWan_CBA',
                                                'Incorrect_Transmission_Power_Settings_LoraWan_CBA',
                                                'Incorrect_coding_rate_CBA', 'Excessive_payload_size_CBA',
                                                'Incorrect_payload_length_CBA', 'Buffer_Overflow_CBA',
                                                'Subscription_Handling_Error_CBA', 'Delayed_Message_Processing_CBA',
                                                'Security_Breach_CBA', 'Inaccurate_Signal_Strength_CBA',
                                                'Incorrect_Distance_Measurement_CBA', 'Signal_Fading_CBA',
                                                'Dynamic_Obstacles_CBA', 'Incorrect_Number_of_Rounds_CBA',
                                                'Incorrect_Number_of_Motes_CBA', 'Incorrect_Activity_Probability_CBA',
                                                'Incorrect_Upper_Bound_Value_CBA', 'Incorrect_Lower_Bound_Value_CBA'],
                 ['Incorrect_region_mapping_LA', 'Incorrect_power_settings_mote_LA', 'Latency_in_Data_Update_LA', 'Inappropriate_Spreading_Factor_Mote_LA', 'Improper_Sampling_Rate_Mote_LA', 'High_Movement_Speed_Mote_LA', 'Incorrect_Power_Threshold_Mote_LA', 'Incorrect_Power_Setting_Gateway_LA', 'Inappropriate_Spreading_Factor_Gateway_LA', 'Incorrect_Power_Threshold_Gateway_LA', 'Incorrect_Shadow_Fading_Setting_LA', 'Incorrect_Path_Loss_Exponent_LA', 'Incorrect_Reference_Distance_LA', 'Incorrect_Signal_Strength_Threshold_LA', 'Incorrect_Distance_Measurement_LA', 'Incorrect_Activity_Probability_LA', 'Inappropriate_number_of_rounds_LA', 'Incorrect_Upper_Bound_Setting_LA', 'Incorrect_Lower_Bound_Setting_LA', 'Incorrect_Simulation_Parameters_LA', 'Incorrect_Cumulative_Results_LA', 'Incorrect_Bandwidth_Settings_of_LoraWan_LA', 'Improper_Spreading_Factor_of_LoraWan_LA', 'Incorrect_Transmission_Power_of_LoraWan_LA', 'Incorrect_Coding_Rate_of_LoraWan_LA', 'Excessive_Payload_Size_of_LoraWan_LA', 'Incorrect_Payload_Length_of_LoraWan_LA', 'Server_Downtime_LA', 'Incorrect_Quality_of_Service_Settings_LA', 'Network_Congestion_LA', 'Security_Breaches_LA', 'Incorrect_adaptation_strategy_logic_LA', 'Incorrect_Input_Profiles_LA', 'Misconfigured_adaptation_goals_intervals_LA', 'Simulation_Logic_Errors_LA', 'Incorrect_Configuration_Settings_LA', 'File_Corruption_Or_Unreadable_File_Format_LA', 'Inaccurate_Input_Profiles_Data_LA', 'Missing_Or_Incomplete_Input_Profiles_LA', 'Simulation_Run_Data_Corruption_LA'],
                 ['MQTT_Protocol_Failure_CSA', 'LoRaWaN_Protocol_Failure_CSA', 'Incorrect_Protocol_Configuration_CSA', 'High_Data_Volumes_Causing_Network_Congestion_CSA', 'Data_Packet_Loss_CSA', 'Insufficient_Data_Handling_Capacity_CSA']],
        'bounds': [[[1, 1000]] * 33,  # For ComponentBasedRIMOO
                  [[1, 18]] * 40,  # For LayeredRIMOO
                  [[1, 18]] * 6] # For ClientServerRIMOO
    }

    # Run the sensitivity analysis for each architecture
    for arch, num_vars, names, bounds, arch_name in zip(architectures, params['num_vars'], params['names'], params['bounds'], architecture_names):
        print(f"Running sensitivity analysis for {arch_name} architecture...")

        # Update problem_param for current architecture
        problem_param = {
            'num_vars': num_vars,
            'names': names,
            'bounds': bounds
        }

        param_values = sobol.sample(problem_param, 1024)

        # Run model (NSGA2 optimization) for each set of sampled parameter values
        Y = np.zeros([param_values.shape[0]])
        for i, X in enumerate(param_values):
            problem = arch()
            algorithm = NSGA2(pop_size=100)
            res = minimize(problem=problem, algorithm=algorithm, termination=('n_gen', 100), seed=1, verbose=False)
            F = res.F
            # Taking the mean of 'Energy Consumption' as the output for evaluation
            Y[i] = np.mean(F[:, 0])

        # Perform sensitivity analysis
        Si = sobol.analyze(problem_param, Y)

        print(f"Sensitivity indices for {arch_name}:", Si['S1'])

    print("Sensitivity analysis completed.")


def main():
    architectures = ['Component-Based', 'Layered', 'Client-Server']
    results = {}
    colors = {'Component-Based': 'blue', 'Layered': 'green', 'Client-Server': 'red'}
    best_solutions = []

    for architecture in architectures:
        time.sleep(10)

        if architecture == 'Component-Based':
            problem = ComponentBasedRIMOO()
        elif architecture == 'Layered':
            problem = LayeredRIMOO()
        elif architecture == 'Client-Server':
            problem = ClientServerRIMOO()

        algorithm = NSGA2(pop_size=100)
        res = minimize(problem=problem, algorithm=algorithm, termination=('n_gen', 500), seed=1, save_history=True,
                       verbose=True)
        results[architecture] = res

    plt.figure(figsize=(10, 6))

    for architecture in architectures:
        if results[architecture] is None or results[architecture].F is None:
            print(f"No result for {architecture}")
            continue
        F = results[architecture].F
        plt.scatter(F[:, 0], F[:, 1], c=colors[architecture], label=architecture)

        distances = np.sqrt((F[:, 0] - F[:, 0].min()) ** 2 + (F[:, 1].max() - F[:, 1]) ** 2)
        best_solution_index = np.argmin(distances)

        best_solutions.append((architecture, F[best_solution_index, 0], F[best_solution_index, 1]))

        print(
            f'Best solution for architecture {architecture} has energy consumption {F[best_solution_index, 0]} and reliability {F[best_solution_index, 1]}.')

        plt.scatter(F[best_solution_index, 0], F[best_solution_index, 1], c='yellow', marker='x')

    if not best_solutions:
        print("No best solutions found for any architecture. Exiting...")
        return

    ideal_point = [0, 1]  # Ideal point (min Energy, max reliability)

    min_distance = np.inf
    most_suitable_architecture = None
    for architecture, energy, reliability in best_solutions:
        distance = np.sqrt((ideal_point[0] - energy) ** 2 + (ideal_point[1] - reliability) ** 2)
        if distance < min_distance:
            min_distance = distance
            most_suitable_architecture = (architecture, energy, reliability)

    print("\n")

    for architecture, energy, reliability in best_solutions:
        print(
            f"Best solution for architecture {architecture} has energy consumption {energy} and reliability {reliability}.")
        if (most_suitable_architecture[0] == architecture):
            print(
                f"The most suitable architecture is {most_suitable_architecture[0]} with energy consumption {most_suitable_architecture[1]} and reliability {most_suitable_architecture[2]}\n")

    plt.xlabel("Energy Consumption")
    plt.ylabel("Reliability")
    plt.title("Pareto Front for Different Architectures")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    run_sensitivity_analysis()