from core_logic import create_agent_instance, override_agent_instance, play_game, TrainingParadigms
import os
import pickle
import yaml
import time
import argparse

paradigms = set('ppo_resnet')

def main():
    parser = argparse.ArgumentParser(
        description="""
Flexible Chess AI Training and Evaluation Framework.

This framework provides tools to train, evaluate, and play against neural network-based chess agents.
It supports modular agent design, instance management, multiple training paradigms, and interactive play.

Typical usage includes:
    - Creating new agent instances with custom configurations (e.g. MCTS guided PPO using ResNet)
    - Training agents using different paradigms (e.g. recursive self-play, combination of Value-Policy loss and PPO)
    - Listing available agents, instances, and associated configurations
    - Playing interactively against a trained model

Example usage:
    python main.py --agent ppo_resnet --create (create a new instance of the ppo_resnet agent)
    python main.py --agent ppo_resnet --instance ppo_resnet_1 --train --paradigm mcts_guided_ppo_self_play (train the ppo_resnet_1 instance using the mcts_guided_ppo_self_play paradigm)
    python main.py --vs-human ppo_resnet_1 (play against the trained model interactively)
        """,
        epilog="""
Additional Notes:
    - Paradigms define specific training strategies. Use --paradigm to specify one.
    - Instances are saved under agent/<agent_name>/<instance_name>/ and contain weights, configs, etc.
    - Playing against a model uses the latest saved weights in the specified instance directory.
    - To modify or inspect configs, edit or view config.yaml inside the instance directory.

Developed for research and experimentation with reinforcement learning in games like Chess.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --agent and --instance arguments
    parser.add_argument('--agent', type=str, help='Name of the agent (e.g. ppo_resnet)')
    parser.add_argument('--instance', type=str, help='Name of the instance (e.g. ppo_resnet_1)')
    
    # --list argument
    parser.add_argument('--list', action='store_true', help=(
        "If no --agent is specified, lists all agents.\n"
        "If --agent is specified and no --instance is specified, lists all instances of the agent.\n"
        "If --agent and --instance are both specified, list configs for the instance."))

    # --play argument
    parser.add_argument('--create', action='store_true', help='Create a new agent instance')
    parser.add_argument('--vs-human', action='store_true', help='Play against a trained model interactively')
    parser.add_argument('--train', action='store_true', help='Train an agent')
    parser.add_argument('--paradigm', type=str, help='Name of the paradigm to use for training')

    args = parser.parse_args()

    # --list command logic
    if args.list:
        if args.paradigm:
            for p in paradigms:
                print(f"- {p}")
            return

        if not args.agent:
            agents = [
                d for d in os.listdir('agent')
                if os.path.isdir(os.path.join('agent', d))
            ]
            print("Available agents:")
            for a in sorted(agents):
                print(f"- {a}")
            return
        
        agent_path = os.path.join('agent', args.agent)
        if not os.path.isdir(agent_path):
            print(f"Agent **{args.agent}** not found.")
            return
        
        if not args.instance:
            instances = [
                d for d in os.listdir(agent_path)
                if os.path.isdir(os.path.join(agent_path, d))
            ]
            print(f"Available instances of agent **{args.agent}**:")
            for i in sorted(instances):
                print(f"- {i}")
            return

        instance_path = os.path.join(agent_path, args.instance)
        if not os.path.isdir(instance_path):
            print(f"Instance **{args.instance}** not found.")
            return
        
        config_path = os.path.join(instance_path, 'config.yaml')
        print(f"Configuration for instance **{args.instance}**:")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                print(yaml.dump(config, sort_keys=False))
        else:
            print(f"No configuration found for instance **{args.instance}**.")

    # --create command logic
    if args.create:
        if not args.agent:
            print("Please specify an agent to create an instance for.")
            return
        
        agent_instance, agent_dir = create_agent_instance(args.agent)
        print(f"Created new instance of agent **{args.agent}** at **{agent_dir}**.")
        return

    # --train command logic
    if args.train:
        if not args.agent or not args.instance or not args.paradigm:
            print("Please specify both --agent and --instance to train an agent.")
            return
        
        instance_path = os.path.join('agent', args.agent, args.instance)
        if not os.path.isdir(instance_path):
            print(f"Instance **{args.instance}** not found.")
            return
    
        with open(os.path.join(instance_path, 'instance.pkl'), 'rb') as f:
            agent_instance = pickle.load(f)
        
        trainer = TrainingParadigms(args.instance)
        paradigm = _retrieve_paradigm(args.paradigm)

        print(f"Running training paradigm <{args.paradigm}> for agent <{args.agent}> instance <{args.instance}>.")
        paradigm(agent_instance)
        print(f"Training completed.")
        return

    # --vs-human command logic
    if args.vs_human:
        if not args.agent or not args.instance:
            print("Please specify both --agent and --instance to play against a trained model.")
            return
        
        instance_path = os.path.join('agent', args.agent, args.instance)
        if not os.path.isdir(instance_path):
            print(f"Instance **{args.instance}** not found.")
            return
        
        with open(os.path.join(instance_path, 'instance.pkl'), 'rb') as f:
            agent_instance = pickle.load(f)
        
        print(f"Playing against trained model **{args.instance}** of agent **{args.agent}**.")
        play_game(agent_instance)
        return

    # --vs-human command logic
    if args.vs_human:
        if not args.agent:
            print("Please specify an agent to play against.")
            return
        
        instance_path = os.path.join('agent', args.agent, args.instance)
        if not os.path.isdir(instance_path):
            print(f"Instance **{args.instance}** not found.")
            return
        
        with open(os.path.join(instance_path, 'instance.pkl'), 'rb') as f:
            agent_instance = pickle.load(f)
        
        print(f"Playing against trained model **{args.instance}** of agent **{args.agent}**.")
        play_game(agent_instance)
        return

def _retrieve_paradigm(trainer, paradigm_name):

    if not hasattr(trainer, paradigm_name):
        print(f"Training paradigm <{paradigm_name}> not found.")
        print("Available paradigms:")
        for m in dir(trainer):
            if not m.startswith('_') and callable(getattr(trainer, m)):
                print(f"- {m}")
        exit(1)
    return getattr(trainer, paradigm_name)


if __name__ == "__main__":
    main()
