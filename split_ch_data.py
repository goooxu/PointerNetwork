import numpy
import math

input_filename = 'data/ch_all.txt'
output_training_filename = 'data/ch_all_training.txt'
output_test_filename = 'data/ch_all_test.txt'

category_length = 20000
category_trainning_length = math.floor(category_length * 0.95)
maximum_point_number = 50

dataset = dict([(k, []) for k in range(5, maximum_point_number + 1)])

fp_input = open(input_filename, 'r')

for line in fp_input:
    columns = line.split()
    point_number = columns.index('output') // 2
    if point_number <= maximum_point_number:
        dataset[point_number].append(line)

perm = numpy.random.permutation(category_length)
training_perm = perm[:category_trainning_length]
test_perm = perm[category_trainning_length:]

fp_output_training = open(output_training_filename, 'w')
fp_output_test = open(output_test_filename, 'w')

for point_number in dataset:
    data = numpy.array(dataset[point_number])
    for line in data[training_perm]:
        fp_output_training.write(line)
    for line in data[test_perm]:
        fp_output_test.write(line)

