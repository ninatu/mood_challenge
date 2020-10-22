import argparse
import yaml
import time

import mood.dpa.train
import mood.dpa.evaluate
import mood.dpa.inference_evaluate_3d


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, choices=['train', 'evaluate', 'inference_evaluate_3d'])
    parser.add_argument('config', type=str, help='Path to config')

    args = parser.parse_args()

    action = args.action
    config_path = args.config

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    model_type = config['model_type']
    if model_type == 'dpa':
        if action == 'train':
            mood.dpa.train.main(config)
        elif action == 'evaluate':
            mood.dpa.evaluate.main(config)
        elif action == 'inference_evaluate_3d':
            mood.dpa.inference_evaluate_3d.main(config)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    print(f"Finished. Took: {(time.time() - start_time) / 60:.02f}m")
