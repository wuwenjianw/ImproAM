Multi-Robot Task Allocation under Time Constraints using Improved Capsule Graph Attention Networks
(ImproCapAM)

How to use the code:

    Training:
        To start training run the run.py file. The options.py can be used for different environement settings such as number of taks, number of robots
        and other parameters.
        The trained models for each epoch will be stored in a directory named 'outputs'
        We recommend using a GPU for training.

    Evaluation:
        The datasets for testing can be found inside the directory named 'data'. Run the eval.py file with the path to the dataset, specify the number of tasks,
        and the number of agents.