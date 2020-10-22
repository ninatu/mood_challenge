import argparse
import yaml
import time

from mood.dpa import train, evaluate, inference_evaluate_3d

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['train', 'evaluate', 'inference_evaluate_3d'])
    parser.add_argument('config', type=str, help='Path to config')

    args = parser.parse_args()

    action = args.action
    config_path = args.config

    with open(args.config[0], 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    if action == 'train':
        train.main(config)
    elif action == 'evaluate':
        evaluate.main(config)
    elif action == 'inference_evaluate_3d':
        inference_evaluate_3d.main(config)
    else:
        raise NotImplementedError()

    print(f"Finished. Took: {(time.time() - start_time) / 60:.02f}m")
