from experiment import GraphPropertyPredictionExperiment
import utils

def main():
    args = utils.parse_args()
    print(args)
    experiment = GraphPropertyPredictionExperiment(args)
    experiment.run()


if __name__ == "__main__":
    main()