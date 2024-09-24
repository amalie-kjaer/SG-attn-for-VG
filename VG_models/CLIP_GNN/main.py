import argparse
from VG_models.CLIP_GNN.trainer import train_model, test_model
from utils import *

def main(args):
    if args.train:
        train_model(args.config_path)
    elif args.test:
        test_model(args.config_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--test", action="store_true", help="Run testing mode")
    parser.add_argument("--resume_from_ckpt", help="Path to latest checkpoint the training should be resumed from")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file")
    
    args = parser.parse_args()
    
    if args.train and args.test:
        parser.error("Only specify one of --train or --test")
    if not args.train and not args.test:
        parser.error("One of --train or --test must be specified")

    main(args)