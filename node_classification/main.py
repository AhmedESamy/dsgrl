import experiment
import utils
import tune

import torch

torch.manual_seed(0)


def evaluate_embedding(embeddings, y, train_mask, val_mask, test_mask, task):
    
    num_classes = y.unique().shape[0] if task.lower() in {"bc", "mcc"} else y.shape[1]
    evaluator = experiment.LinearEvaluationExperiment(
        in_dim=embeddings.shape[1], out_dim=num_classes, device=embeddings.device, 
        task=task, epochs=500) #epochs = 500 for reddit and yelp.
    evaluator.execute(
        x=embeddings, y=y, train_mask=train_mask, 
        val_mask=val_mask, test_mask=test_mask)


def main():
    args = utils.parse_args()
    print(args)
    if args.tune:
        tune.tune(args)
    else:
        assert args.task in {'bc', 'mcc', 'mlc'}

        utils.log(args, verbose=args.verbose)
    
        # Self-supervised training
        exp = experiment.NodeClassificationExperiment(args=args)
        
        data = exp.dataset.data
        
        embeddings = exp.infer_embedding()
        # Evaluation before training
        evaluate_embedding(
            embeddings, y=data.y, train_mask=data.train_mask, 
            val_mask=data.val_mask, test_mask=data.test_mask, task=args.task)
        
        exp.run()

        exp.pause_training_mode()
        if args.aug_type == "topology":
            exp.save_hon()
        torch.cuda.empty_cache()
        num_classes = int(exp.dataset.num_classes)

        embeddings = exp.infer_embedding()
        # Evaluation after training
        evaluate_embedding(
            embeddings, y=data.y, train_mask=data.train_mask, 
            val_mask=data.val_mask, test_mask=data.test_mask, task=args.task)
  

if __name__ == "__main__":
    main()
