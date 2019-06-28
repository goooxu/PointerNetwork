The pointer network with beam search is used to solve such below problems.

## 1st: MWT problem
The full name of MWT is Minimum-weight triangulation, could refer to the wikipedia [page](https://en.wikipedia.org/wiki/Minimum-weight_triangulation).

### Download dataset
The dataset can be visited from [here](http://goo.gl/NDcOIG), please download the `convex_hull_5-50_train.txt.zip` file and extract it to `data` folder, and rename it to `ch_all.txt`

### Split training and test data
Run `python3 split_ch_data.py` to split training and test data, you can set the `max_point_number` inside the source code, the default value is `50`, which means take the point number from 5 to 50 from the whole dataset.

### Training and test
Create `models` folder for saving models, then run `python3 train_ch.py` to start training, also you can adjust some parameter inside the source code.