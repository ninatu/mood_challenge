import argparse
import yaml
import time

import anomaly_detection.dpa.train
import anomaly_detection.dpa.evaluate
import anomaly_detection.dpa.inference_evaluate_3d


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
            anomaly_detection.dpa.train.main(config)
        elif action == 'evaluate':
            anomaly_detection.dpa.evaluate.main(config)
        elif action == 'inference_evaluate_3d':
            anomaly_detection.dpa.inference_evaluate_3d.main(config)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    print(f"Finished. Took: {(time.time() - start_time) / 60:.02f}m")
