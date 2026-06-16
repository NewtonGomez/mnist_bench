# MNIST Bench

Experimental comparison between a baseline multilayer neural network and a support vector machine for handwritten digit classification on the MNIST dataset.

---

## Project Description

This project aims to compare the performance of two supervised classification models applied to the MNIST dataset:

* A baseline multilayer neural network, implemented as **NNC / MLP**.
* A multi-class support vector machine, implemented via a **SVC One-vs-One** strategy.

The experiment analyzes the behavior of both models through training curves, performance metrics, cross-validation partitions, and statistical hypothesis testing.

---

## Hypothesis

The primary hypothesis of the project is:

> While both models can perform efficiently in classifying handwritten digits, the baseline neural network will achieve superior performance compared to the support vector machine.

Formally defined as:

* $H_0: \text{mean}(\text{Accuracy}_{\text{NNC}} - \text{Accuracy}_{\text{SVC}}) \le 0$
* $H_1: \text{mean}(\text{Accuracy}_{\text{NNC}} - \text{Accuracy}_{\text{SVC}}) > 0$

Where $\text{Accuracy}_{\text{NNC}}$ represents the final accuracy of the neural network and $\text{Accuracy}_{\text{SVC}}$ represents the final accuracy of the support vector machine evaluated across the same cross-validation folds.

---

## Dataset

The project utilizes the **Modified National Institute of Standards and Technology database (MNIST)**.

MNIST contains grayscale images of handwritten digits from 0 to 9. Each image has a resolution of **28 x 28** pixels, meaning each instance is represented as a **784**-feature vector after flattening.

The standard version of the dataset contains:

* **60,000** training images
* **10,000** testing images
* **10** classes: digits from 0 to 9

The target variable is nominal categorical, representing the corresponding digit for each image matrix.

---

## Motivation

MNIST was selected because it is a widely known, well-documented, and standard benchmark dataset within the machine learning community. Furthermore, it served as an established baseline during project development, allowing the analytical focus to remain entirely on the comparative methodology between models rather than investing additional timeline resources into initial domain engineering.

---

## Evaluated Models

### NNC / MLP

The neural network used corresponds to a **Multilayer Perceptron (MLP)** architecture featuring fully connected dense layers, ReLU activation functions, dropout regularization, and a ten-neuron output layer.

The network is trained minimizing cross-entropy loss via gradient-based optimization. The implementation is built using **MLX**, an array framework optimized explicitly for Apple Silicon architecture.

### SVC One-vs-One

The support vector machine is deployed as a multi-class framework utilizing the **One-vs-One** strategy. For a 10-class problem, this strategy trains:

$$\frac{10 \times (10 - 1)}{2} = 45 \text{ binary classifiers}$$

The ensemble's final classification decision is reached via majority voting across all 45 binary estimators.

---

## Performance Metrics

The primary evaluation metric utilized is:

* **Accuracy**

Historical logs of the following metrics are also recorded for both frameworks across training checkpoints to generate comparative tracking trajectories:

* `loss`
* `accuracy`

The accuracy metric computation is evaluated via the `accuracy_score` function from `sklearn.metrics`.

---

## Experimental Validation

Model performance comparison is executed through stratified cross-validation. Both structural models are trained and tested utilizing the exact same evaluation folds to ensure a strictly paired experimental design.

The layout tracks output data fold-by-fold:


$$\text{fold} \mid \text{accuracy\_nnc} \mid \text{accuracy\_svc} \mid \text{difference}$$

The paired difference per partition is computed as:


$$d_i = \text{Accuracy}_{\text{NNC}_i} - \text{Accuracy}_{\text{SVC}_i}$$

---

## Hypothesis Testing

To evaluate the primary research statement, a one-tailed paired t-test is applied:

* $H_0: \text{mean}(d) \le 0$
* $H_1: \text{mean}(d) > 0$

The non-parametric **Wilcoxon signed-rank test** for paired samples is additionally evaluated as a complementary robust verification check. The statistical reports include:

* Mean difference.
* Median of the differences.
* Paired standard deviation.
* Standardized effect size via paired Cohen's d.
* 95% empirical bootstrap confidence interval.

---

## Main Results

The baseline experiment was evaluated using 10 paired validation folds. The metrics obtained were:

| Metric | NNC / MLP | SVC One-vs-One | Difference (NNC - SVC) |
| --- | --- | --- | --- |
| **Final Accuracy Mean** | 0.967200 | 0.881600 | 0.085600 |
| **Standard Deviation** | 0.005953 | 0.008631 | 0.007311 |
| **Median** | 0.968333 | 0.879667 | 0.086667 |
| **Minimum** | 0.954667 | 0.868667 | 0.073333 |
| **Maximum** | 0.973333 | 0.898667 | 0.094000 |
| **Paired Folds** | 10 | 10 | 10 |

The neural network achieved a consistent average accuracy advantage of approximately **8.56 percentage points** over the One-vs-One SVC model.

### Statistical Results

* **Paired t-test:** $t\text{-statistic} = 37.024714$, $p\text{-value} < 0.001$
* **Wilcoxon test:** $\text{statistic} = 55.0$, $p\text{-value} = 0.0009765625$

In both test environments, the null hypothesis is rejected at a significance level of $\alpha = 0.05$.

The 95% empirical bootstrap confidence interval calculated for the mean performance gap was:


$$[0.081200, 0.089733]$$

Since the entire confidence interval bounds sit strictly above zero, the performance margin favors the neural network architecture across all resamples.

---

## General Conclusion

The empirical findings strongly support the project's hypothesis. While both learning models demonstrate efficient classification capabilities handling handwritten digits on MNIST, the **NNC / MLP** network delivers structurally superior generalization performance compared to the **SVC One-vs-One** model in terms of final accuracy.

The performance advantage of the deep neural network is validated by:

* Higher overall average accuracy.
* Faster and smoother convergence on training curves.
* Lower structural final loss values.
* Higher distribution boundaries across validation partitions.
* Highly significant statistical evidence from paired hypothesis checks.

---

## General Project Structure

```text
mnist_bench/
├── data/
│   └── raw/
├── results/ # not uploated saving space
├── scripts/
│   ├── benchmark_mnist_folds.py
│   ├── hypothesis_test_mnist.py
│   ├── plot_histories.py
│   └── plot_mnist_overview.py
├── src/
│   ├── models/
│   │   ├── mlp.py
│   │   └── svm.py
│   └── tools.py
└── README.md

```

---

## Main Scripts

### Run Benchmark with Folds

```bash
python3 scripts/benchmark_mnist_folds.py

```

Trains both classification models, runs the cross-validation partitions, and saves training execution logs.

### Run Hypothesis Test

```bash
python3 scripts/hypothesis_test_mnist.py

```

Imports the saved fold histories to calculate parametric and non-parametric statistical metrics.

### Generate Comparative Plots

```bash
python3 scripts/plot_histories.py

```

Renders loss curves, accuracy curves, and final validation performance distribution boxplots.

### Generate Visual MNIST Exploration

```bash
python3 scripts/plot_mnist_overview.py

```

Generates an exploratory overview plot detailing class instances, random image samples, and mean archetype digits.

---

## Expected Outputs

The codebase automatically creates and exports files to the following destinations:

* `results/histories/nnc_fold_01.npz`
* `results/histories/svc_fold_01.npz`
* `results/plots/accuracy_comparison.png`
* `results/plots/loss_comparison.png`
* `results/plots/final_accuracy_boxplot.png`
* `results/plots/mnist_overview.png`
* `results/statistics/mnist_hypothesis_test_report.txt`
* `results/statistics/mnist_hypothesis_test_summary.json`
* `results/statistics/mnist_hypothesis_test_by_fold.csv`

---

## Requirements

This project requires Python 3 and the following primary libraries:

```bash
pip install numpy scikit-learn scipy matplotlib pandas mlx

```

> **Note:** The MLX array engine framework is designed and optimized explicitly for Apple Silicon hardware execution.

---

## References

* LeCun, Y., Cortes, C. & Burges, C. J. C. *The MNIST Database of Handwritten Digits.*
* LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. *Gradient-Based Learning Applied to Document Recognition.*
* Cortes, C. & Vapnik, V. *Support-Vector Networks.*
* Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.*
* Apple Machine Learning Research. *MLX: An Array Framework for Apple Silicon.*
* SciPy Developers. *scipy.stats.ttest_rel.*