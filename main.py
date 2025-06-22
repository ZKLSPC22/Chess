import os
import inquirer
import logging
import yaml
from utils.config import override_config
from utils.logger import setup_loggers
from core_logic import DataCollectionParadigms, MixedParadigm
from core_logic import create_agent_instance, vs_human, vs_agent_with_render, TrainingParadigms, load_instance


paradigms = set(['ppo_self_play'])

def main():
    # 1. Set up loggers first, using a temporary config.
    # We will load the full config right after.
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        setup_loggers(config)
    except FileNotFoundError:
        # Fallback if config.yaml doesn't exist yet
        print("ERROR: config.yaml not found. Cannot set up logging.")
        config = {'logging': {}}
        setup_loggers(config)

    try:
        logging.info("Application starting.")
        
        # Mode selection
        mode = inquirer.prompt([
            inquirer.List('mode', message="Select a mode", choices=['train', 'vs_human', 'vs_agent', 'create_agent'])
        ])['mode']
        logging.info(f"Running in '{mode}' mode.")

        # Centralized config resolution
        agent_name = None
        final_agent_config = None

        if mode != 'create_agent':
            agent_name = select_agent()
            
            # Build the agent's final configuration from the central config file
            defaults = config.get('default_configs', {})
            agent_specifics = config.get('agent_settings', {}).get(agent_name, {})
            final_agent_config = override_config(agent_specifics, defaults)

        if mode == 'train':
            instance_dir = select_instance(agent_name)
            agent_instance = load_instance(instance_dir, final_agent_config)
            paradigm_executor = select_paradigm(agent_instance, instance_dir)
            
            paradigm_executor()

        elif mode == 'vs_human':
            instance_dir = select_instance(agent_name)
            agent_instance = load_instance(instance_dir, final_agent_config)
            vs_human(agent_instance)

        elif mode == 'vs_agent':
            print("Select Agent 1 (White):")
            # Agent 1 config resolution is already done for `agent_name`
            instance1_dir = select_instance(agent_name)
            agent1_instance = load_instance(instance1_dir, final_agent_config)

            print("Select Agent 2 (Black):")
            agent2_name = select_agent()
            
            # Build Agent 2's config
            defaults2 = config.get('default_configs', {})
            agent_specifics2 = config.get('agent_settings', {}).get(agent2_name, {})
            final_agent_config2 = override_config(agent_specifics2, defaults2)

            instance2_dir = select_instance(agent2_name)
            agent2_instance = load_instance(instance2_dir, final_agent_config2)

            vs_agent_with_render(agent1_instance, agent2_instance)

        elif mode == 'create_agent':
            agent_name = select_agent()
            # Build config for the new agent
            defaults = config.get('default_configs', {})
            agent_specifics = config.get('agent_settings', {}).get(agent_name, {})
            final_agent_config = override_config(agent_specifics, defaults)
            create_agent_instance(agent_name, final_agent_config)
        
        logging.info("Application finished successfully.")

    except Exception as e:
        logging.critical("An unhandled exception occurred in the main execution block.", exc_info=True)
        raise


def select_agent(prompt_message="Select an agent"):
    agent_dirs = [d for d in os.listdir('agent') if os.path.isdir(os.path.join('agent', d)) and not d.startswith('__') and d != 'utils']
    if not agent_dirs:
        logging.warning("No agent directories found in the 'agent' folder.")
        print("No agents found. Please create an agent directory first.")
        exit()
    questions = [inquirer.List('agent', message=prompt_message, choices=agent_dirs)]
    answers = inquirer.prompt(questions)
    logging.info(f"User selected agent: {answers['agent']}")
    return answers['agent']

def select_instance(agent_name):
    agent_dir = os.path.join('agent', agent_name)
    if not os.path.exists(agent_dir):
        logging.error(f"Agent directory does not exist: {agent_dir}")
        print(f"Error: Agent directory '{agent_name}' not found.")
        exit()
    instance_dirs = [d for d in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, d)) and d.startswith('instance')]
    
    choices = instance_dirs + ["Create new instance"]
    questions = [inquirer.List('instance', message="Select an instance", choices=choices)]
    answers = inquirer.prompt(questions)
    
    instance_choice = answers['instance']
    logging.info(f"User selected instance: {instance_choice}")

    if instance_choice == "Create new instance":
        # This path is now handled inside the 'create_agent' mode in main()
        # We need to resolve the config before creating
        print("Please use the 'create_agent' mode from the main menu to create a new instance.")
        exit()
    else:
        return os.path.join(agent_dir, instance_choice)

def select_paradigm(agent_instance, instance_dir):
    training_paradigms = TrainingParadigms(agent_instance, instance_dir)
    paradigm_names = training_paradigms.list_paradigms()
    if not paradigm_names:
        logging.error(f"No training paradigms found for agent: {agent_instance.__class__.__name__}")
        print("Error: No training paradigms available for this agent.")
        exit()
    questions = [inquirer.List('paradigm', message="Select a training paradigm", choices=paradigm_names)]
    answers = inquirer.prompt(questions)
    selected_paradigm = answers['paradigm']
    logging.info(f"User selected paradigm: {selected_paradigm}")
    return getattr(training_paradigms, selected_paradigm)

if __name__ == "__main__":
    main()
