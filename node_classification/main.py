import experiment
import utils
import tune

import torch

torch.manual_seed(0)


def main():
    args = utils.parse_args()
    print(args)
    if args.tune:
        tune.tune(args)
    else:
        assert args.task in {'bc', 'mcc', 'mlc'}
        utils.log(args, verbose=args.verbose)
        
        exp = experiment.NodeClassificationExperiment(args=args)
        
        print("\n======================Before Self-Supervised Training=============================\n")
        
        exp.eval_model(test=True)
        
        print("\n==========================Self-Supervised Training================================\n")
        
        exp.run()

        if args.aug_type == "topology":
            exp.save_hon()
            
        print("\n=======================After Self-Supervised Training=============================\n")
        
        torch.cuda.empty_cache()
        print(exp.best_score_tracker)
        print()
        
        exp.finish()
        print()

if __name__ == "__main__":
    main()
