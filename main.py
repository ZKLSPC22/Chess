import os
import inquirer
import logging
import yaml
from utils.config import override_config
from utils.logger import setup_loggers
from core_logic import create_agent_instance, vs_human, vs_agent_with_render, TrainingParadigms, load_instance

BACK_CHOICE = '\x1b[94m<-- Back\x1b[0m'  # Light blue color for 'Back' option

def prompt_with_back(question):
    """Wrapper for inquirer.prompt that adds a 'Back' option and handles it."""
    # Check if 'choices' is a key in the question dictionary
    if 'choices' in question:
        # Make sure choices is a list before appending
        if isinstance(question['choices'], list):
            question['choices'].append(BACK_CHOICE)
    
    # The dictionary is converted to an inquirer.List object before being passed
    # to inquirer.prompt(). This fixes the AttributeError.
    list_question = inquirer.List(
        name=question['name'],
        message=question['message'],
        choices=question['choices']
    )
    answers = inquirer.prompt([list_question])

    if not answers:  # User pressed Ctrl+C
        logging.info("User cancelled interaction. Exiting.")
        exit()
    
    # Extract the answer, assuming there is only one question
    answer_key = question['name']
    if answers[answer_key] == BACK_CHOICE:
        return '__BACK__'
    
    return answers[answer_key]

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
        
        state = 'MODE_SELECTION'
        while state != 'EXIT':
            if state == 'MODE_SELECTION':
                mode_question = inquirer.List(
                    'mode',
                    message="Select a mode",
                    choices=['train', 'vs_human', 'vs_agent', 'create_agent', 'Quit']
                )

                # We don't want a "back" option here. Quit is the way out.
                answers = inquirer.prompt([mode_question])
                if not answers:
                    state = 'EXIT'
                    continue
                
                mode = answers['mode']
                logging.info(f"Running in '{mode}' mode.")

                if mode == 'Quit':
                    state = 'EXIT'
                else:
                    state = mode.upper() # e.g., 'TRAIN', 'VS_HUMAN'

            elif state == 'TRAIN':
                # Training Logic
                agent_name = select_agent()
                if agent_name == '__BACK__':
                    state = 'MODE_SELECTION'
                    continue
                
                defaults = config.get('default_configs', {})
                agent_specifics = config.get('agent_settings', {}).get(agent_name, {})
                final_agent_config = override_config(agent_specifics, defaults)

                instance_dir, agent_instance = select_instance(agent_name, final_agent_config)
                if instance_dir == '__BACK__':
                    state = 'TRAIN' # This is a bit tricky, should go to agent selection.
                                    # For now, let's restart the TRAIN state.
                    continue
                
                if agent_instance is None:
                    agent_instance = load_instance(instance_dir, final_agent_config)

                paradigm_executor = select_paradigm(agent_instance, instance_dir)
                if paradigm_executor == '__BACK__':
                    # This should go back to instance selection.
                    # A full state machine would be better, but for now, let's also restart.
                    state = 'TRAIN'
                    continue
                
                paradigm_executor()
                state = 'MODE_SELECTION' # Go back to main menu after task is done

            elif state == 'VS_HUMAN':
                # Vs Human Logic
                agent_name = select_agent()
                if agent_name == '__BACK__':
                    state = 'MODE_SELECTION'
                    continue
                
                defaults = config.get('default_configs', {})
                agent_specifics = config.get('agent_settings', {}).get(agent_name, {})
                final_agent_config = override_config(agent_specifics, defaults)

                instance_dir, agent_instance = select_instance(agent_name, final_agent_config)
                if instance_dir == '__BACK__':
                    state = 'VS_HUMAN'
                    continue
                
                if agent_instance is None:
                    agent_instance = load_instance(instance_dir, final_agent_config)
                
                vs_human(agent_instance)
                state = 'MODE_SELECTION'

            elif state == 'VS_AGENT':
                # Vs Agent Logic
                print("Select Agent 1 (White):")
                agent1_name = select_agent()
                if agent1_name == '__BACK__':
                    state = 'MODE_SELECTION'
                    continue
                
                defaults1 = config.get('default_configs', {})
                agent_specifics1 = config.get('agent_settings', {}).get(agent1_name, {})
                final_agent_config1 = override_config(agent_specifics1, defaults1)

                instance1_dir, agent1_instance = select_instance(agent1_name, final_agent_config1)
                if instance1_dir == '__BACK__':
                    state = 'VS_AGENT'
                    continue

                if agent1_instance is None:
                    agent1_instance = load_instance(instance1_dir, final_agent_config1)

                print("Select Agent 2 (Black):")
                agent2_name = select_agent()
                if agent2_name == '__BACK__':
                    # Ideally, back to agent 1 instance selection
                    state = 'VS_AGENT'
                    continue

                defaults2 = config.get('default_configs', {})
                agent_specifics2 = config.get('agent_settings', {}).get(agent2_name, {})
                final_agent_config2 = override_config(agent_specifics2, defaults2)
                
                instance2_dir, agent2_instance = select_instance(agent2_name, final_agent_config2)
                if instance2_dir == '__BACK__':
                    # Ideally, back to agent 2 selection
                    state = 'VS_AGENT' # Restart vs_agent flow
                    continue

                if agent2_instance is None:
                    agent2_instance = load_instance(instance2_dir, final_agent_config2)

                vs_agent_with_render(agent1_instance, agent2_instance)
                state = 'MODE_SELECTION'

            elif state == 'CREATE_AGENT':
                # Create Agent Logic
                agent_name = select_agent()
                if agent_name == '__BACK__':
                    state = 'MODE_SELECTION'
                    continue

                defaults = config.get('default_configs', {})
                agent_specifics = config.get('agent_settings', {}).get(agent_name, {})
                final_agent_config = override_config(agent_specifics, defaults)
                create_agent_instance(agent_name, final_agent_config)
                state = 'MODE_SELECTION'
        
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
    
    question = {
        'name': 'agent',
        'message': prompt_message,
        'choices': agent_dirs
    }
    selected_agent = prompt_with_back(question)

    if selected_agent != '__BACK__':
        logging.info(f"User selected agent: {selected_agent}")
        
    return selected_agent

def select_instance(agent_name, final_agent_config):
    agent_dir = os.path.join('agent', agent_name)
    if not os.path.exists(agent_dir):
        logging.error(f"Agent directory does not exist: {agent_dir}")
        print(f"Error: Agent directory '{agent_name}' not found.")
        exit()
    instance_dirs = [d for d in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, d)) and d.startswith('instance')]
    
    choices = instance_dirs + ["Create new instance"]
    question = {
        'name': 'instance',
        'message': "Select an instance",
        'choices': choices
    }
    instance_choice = prompt_with_back(question)

    if instance_choice == '__BACK__':
        return '__BACK__', None

    logging.info(f"User selected instance: {instance_choice}")

    if instance_choice == "Create new instance":
        agent_instance, new_instance_dir = create_agent_instance(agent_name, final_agent_config)
        return new_instance_dir, agent_instance
    else:
        return os.path.join(agent_dir, instance_choice), None

def select_paradigm(agent_instance, instance_dir):
    while True:
        # We need the config for loading opponents.
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Use a dummy instance to get the list of available paradigms.
        # This avoids fully instantiating it before we know if an opponent is needed.
        paradigm_names = TrainingParadigms(agent_instance, instance_dir).list_paradigms()
        
        if not paradigm_names:
            logging.error(f"No training paradigms found for agent: {agent_instance.__class__.__name__}")
            print("Error: No training paradigms available for this agent.")
            exit()
            
        question = {
            'name': 'paradigm',
            'message': "Select a training paradigm",
            'choices': paradigm_names
        }
        selected_paradigm = prompt_with_back(question)

        if selected_paradigm == '__BACK__':
            return '__BACK__'

        logging.info(f"User selected paradigm: {selected_paradigm}")

        opponent_instance = None
        opponent_instance_dir = None

        if selected_paradigm == 'vp_train':
            print("Select opponent agent:")
            opponent_agent_name = select_agent(prompt_message="Select an opponent agent")

            if opponent_agent_name == '__BACK__':
                continue # Go back to paradigm selection

            # Build opponent's config before selecting an instance
            defaults = config.get('default_configs', {})
            agent_specifics = config.get('agent_settings', {}).get(opponent_agent_name, {})
            final_opponent_config = override_config(agent_specifics, defaults)
            
            opponent_instance_dir, opponent_instance = select_instance(opponent_agent_name, final_opponent_config)

            if opponent_instance_dir == '__BACK__':
                continue # Go back to paradigm selection

            if opponent_instance is None:
                opponent_instance = load_instance(opponent_instance_dir, final_opponent_config)
        
        # Now, create the final TrainingParadigms object with all necessary components.
        training_paradigms = TrainingParadigms(
            agent_instance=agent_instance,
            instance_dir=instance_dir,
            opponent_instance=opponent_instance,
            opponent_instance_dir=opponent_instance_dir
        )

        return getattr(training_paradigms, selected_paradigm)

if __name__ == "__main__":
    main()
