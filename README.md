# DND Reimplementation

## 🚀 My schedule

*read and understand the paper*

- [X] prepare the data
  you can get the data at [Huggingface](https://huggingface.co/sanyier312/DND-Reimplementation/tree/main/data/synthetic)
- [X] find out the input and output
- [X] draw the model architecture  flow chart
  ![](20250514213026.png)
- [X] define the loss function

Given a graph $G = (V, E)$, each node $v_i \in V$ has a ground truth label $y_i \in [0, 1]$,
and the model predicts a value $\hat{y}_i \in [0, 1]$.

The Mean Squared Error loss is defined as:

![](20250514215425.png)

- [X] define the evaluation metrics

*coding*

- [X] define the training and evaluation process
- [X] define the hyperparameters
- [X] process the data
- [X] train the model
- [X] evaluate the model
- [X] visualize the results

## 📌 Structure

- `model/` — GAT + MLP model
- `scripts/` — Training and evaluation scripts
- `data/` — Synthetic and real-world directed networks
- `results/` — Logs, plots, metrics

## 🔧 Dependencies

```bash
pip install -r requirements.txt
```

# 📈 Results

*train the model with  ER dataset*

![](er_train.jpg)

*train the model with  SBM dataset*

![](sbm_train.jpg)

*train the model with  SPL dataset*

![](20250516133108.png)

*evaluate the model with  ER dataset compared with baseline*
![](20250523100301.png)

*evaluate the model with  SBM dataset compared with baseline*
![](20250523100422.png)


*evaluate the model with  SPL dataset compared with baseline*
![](20250523100513.png)


*evaluate the model with  wiki-vote dataset compared with baseline*
![](20250523101350.png)

*evaluate the model with  p2p dataset compared with baseline*
![](20250523100853.png)

*evaluate the model with  cit-HepTh dataset compared with baseline*
![](20250523101302.png)

*evaluate the model with  cfinder-google dataset compared with baseline*
![](20250523101054.png)

*evaluate the model with  subelj-cora dataset compared with baseline*
![](20250523101130.png)

*evaluate the model with  cit-HepTh dataset compared with baseline*
![](20250523100936.png)