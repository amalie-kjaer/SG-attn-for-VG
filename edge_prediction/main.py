import argparse
from trainer import train_model, test_model


def main(args):
    if args.mode == "train":
        train_model(args.config_path)
    elif args.mode == "test":
        results = test_model(args.config_path)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()

    main(args)