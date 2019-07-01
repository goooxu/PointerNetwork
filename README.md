# Pointer Networks in PyTorch

This is an implementation of Pointer Networks to solve such problems. Beam search is supported.

## 1st problem: Construct convex hulls
If you don't know what is convex hulls, you could find the definition from [here](https://en.wikipedia.org/wiki/Convex_hull). 

**Generate dataset**

Create `data` folder for saving dataset, then run `node generate_ch_data.js` to generate training and test data, you could adjust some parameters inside the source code.

**Training and test**

Create `models` folder for saving models, then run `python3 train_ch.py` to start training, also you could adjust some parameters inside the source code.

**Review the incorrect cases**

Run `visualize_ch_data.html` in your browser, then copy & paste the incorrect samples to textbox and visualize it.

## 2nd problem: Find MWT
The full name of MWT is minimum-weight triangulation, could refer to the wikipedia [page](https://en.wikipedia.org/wiki/Minimum-weight_triangulation).

## 3rd problem: 中文分词
