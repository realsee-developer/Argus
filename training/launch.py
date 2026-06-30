import argparse
from typing import NoReturn
from hydra import initialize, compose
from trainer import Trainer


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for model training configuration.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - config: Name of the Hydra config file (without .yaml extension)
    """
    parser = argparse.ArgumentParser(
        description="Model Training Entry Point with Hydra Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the Hydra configuration file (without .yaml extension) located in the 'config' directory"
    )
    
    return parser.parse_args()


def main() -> NoReturn:
    """
    Main entry point for model training:
    1. Parses command line arguments
    2. Loads Hydra configuration from specified YAML file
    3. Initializes Trainer with configuration parameters
    4. Executes the training pipeline
    
    Raises:
        Exception: If any error occurs during configuration loading or training execution
    """
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize Hydra and load configuration from config directory
        # version_base=None maintains compatibility with older Hydra versions
        with initialize(version_base=None, config_path="config"):
            config = compose(config_name=args.config)
        
        # Initialize trainer with configuration parameters and start training
        trainer = Trainer(**config)
        trainer.run()
        
    except Exception as e:
        print(f"Error during training execution: {str(e)}")
        raise  # Re-raise exception to preserve stack trace for debugging


if __name__ == "__main__":
    # Execute main training workflow
    main()