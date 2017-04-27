import sys
from pyspark import SparkConf, SparkContext
import numpy
import operator

conf = SparkConf().setAppName("matrix multiply")
sc = SparkContext(conf=conf)
assert sc.version >= '1.5.1'

inputs = sys.argv[1]
output = sys.argv[2]
d = 10

#finding the outer product by tranposing the matrix A
def rdds(mat):
    a = mat.split()
    a = [float(num) for num in a]
    a_transpose = zip(a)
    result_matrix = numpy.multiply.outer(a_transpose, a)
    final_matrix = numpy.matrix(result_matrix)

    return final_matrix


def main():
    text = sc.textFile(inputs)

    result_rdds = text.map(lambda mat: rdds(mat))
    result_data = result_rdds.reduce(operator.add)
    result_matrix_arr = numpy.array(result_data)
    output_list = []
    f = open(output, 'w')
    for row_index in range(len(result_matrix_arr)):
        row = list(result_matrix_arr[row_index])
        row_str = ''
        for col_index in range(d):
            row_str += str(row[col_index]) + ' '
        output_list.append(row_str)
        f.write(row_str)
        f.write("\n")
    f.close()

if __name__ == '__main__':
    main()
