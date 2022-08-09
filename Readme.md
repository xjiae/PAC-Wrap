# Download and unzip datasets
- Please download needed datasets from this link: [datasets](https://drive.google.com/file/d/1KB3vYkYPZQ28b9tPHly4DPbpeiHD4CVJ/view?usp=sharing).
- Unzip the datasets.zip file under the root folder *Implementation*.

# Install needed package
- PyTorch - 1.8.0
- TensorFlow - 1.10.0
- keras - 2.2.4
- scikit-learn - 0.24.2
- scipy - 1.6.2
- 

# Run experiments
- To run Q1 experiments:
-- Run on synthetic dataset and get the statistical results: 
```python
python iid_Q1.py --synth
python analysis.py --synth
```
-- Run on benchmark dataset
```python
python iid_Q1.py
python analysis.py
```
- To run Q2 experiments:
-- Run on i.i.d. *campaign* dataset:
```python
python iid_Q2.py --data_set bank-additional-full_normalised
```
-- Run on i.i.d. *celeba* dataset:
```python
python iid_Q2.py --data_set celeba_baldvsnonbald_normalised
```
-- Run on i.i.d. *census* dataset:
```python
python iid_Q2.py --data_set census-income-full-mixed-binarized
```
-- Run on time series *SMD* dataset:
```python
python ts_Q2.py --data SMD
```
-- Run on time series *NASA* dataset:
```python
python ts_Q2.py --data NASA
```
- To run Q3 experiment:
```python
python iid_Q3.py
```
- To run Q4 experiment:
```python
python iid_Q4.py
```