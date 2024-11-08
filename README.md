# <div align="center">Assignment 1</div>

## <div align="center">CS 451/551</div>
### <div align="center">Fall 2024</div>

## <u>Table of Contents</u>:
- [Install](#install)
- [Implementation](#implementation)
- [Parameter Setting & Run](#parameter-setting--run)
- [Footnotes](#footnotes)
## Install
> **_NOTE:_** This assignment will be tested on [Python 3.10](https://www.python.org/downloads/release/python-31011/)

> Creating a [virtual environment](https://docs.python.org/3.10/library/venv.html) is recommended.

You can install this Python project from scratch by following these steps:

1. Download the whole project from LMS.
2. Install [Python 3.10](https://www.python.org/downloads/release/python-31011/).
3. Install the required libraries via `pip`:
    ```bash
    pip install -r requirements.txt 
    ```
   
## Implementation
You are expected to implement the inside of `GeneticSolver.py` file, especially `solve` method. For a given graph, you will generate a valid and promising solution object. The structure of graph and solution objects are:

- **Graph**: `Dict[str, Set[str]]` indicates the set of adjacent nodes of each node in the graph.
- **Solution**: `Dict[str, str]` denotes the assigned color of each node.

> **_NOTE:_** Colors can be *'Red'*, *'Green'*, *'Yellow'* and *'Blue'*

You are free to utilize any function in `GraphColorProblem.py` file, but any change in that file is forbidden. Please, carefully read the comments before utilizing the methods.

## Parameter Setting & Run
With `main.py` file, you can run & test your implementation as follows:

```bash
python main.py
```

However, in the main file, you need to define:

- File path of target graph ([Pickle](https://docs.python.org/3.10/library/pickle.html) format)
- Entering the best performing selection algorithm (*'Tournament'* or *'Wheel Roulette'* selection) based on your evaluation results.
- Entering the most promising value for each parameter (e.g., population size, max. number of iterations, mutation rate, elitism ratio) after hyperparameter tuning process.

## Footnotes
If you have any questions regarding the homework, we encourage you to seek help during our office hours. Our office hours are held online via Zoom to ensure accessibility for all students, and the corresponding meeting links can be found on LMS. Additionally, if you prefer to communicate via e-mail, you can contact [Anıl Doğru](mailto:anil.dogru@ozu.edu.tr) and [Emre Kuru](mailto:emre.kuru@ozu.edu.tr). Please keep in mind that we cannot provide any extra code or ideas before the deadline, as this would be considered a violation of academic integrity. However, we will do our best to assist you with any conceptual or technical questions you may have.

Please, carefully read the given description.

Good Luck :)