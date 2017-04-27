from pyspark import SparkConf, SparkContext
from scipy.sparse import *
from scipy import *
import sys

#converting row from input file to CSR matrix
def convert_to_csr_matrix(line):
	indices_list = []
	indptr_list = [0]
	data_list = []
	length = len(line)
	indptr_list.append(length)
	for i in range(length):
		x,y = line[i].strip().split(':')
		indices_list.append(int(x))
		data_list.append(float(y))

#CSR row matrix and column matrix is created using transpose

	cmatrix_row = csr_matrix( (array(data_list), array(indices_list), array(indptr_list)),
						shape=(1,100))
	cmatrix_column = cmatrix_row.transpose()

	submatrix = cmatrix_column.multiply(cmatrix_row).todense()

	return submatrix

def main():
	inputs = sys.argv[1]
	output = sys.argv[2]

	conf = SparkConf().setAppName('sparse scalable multiplication')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'

	text = sc.textFile(inputs)
	row = text.map(lambda line: line.split())


	sub = row.map(convert_to_csr_matrix).reduce(lambda a,b: a+b)


	sub_list = sub.tolist()

	result = open(output, 'w')

	for i in range(len(sub_list)):
		for j in range(len(sub_list)):
			if(sub_list[i][j] != 0.0):
				result.write(str(j) + ':' + str(sub_list[i][j]) + " ")
			if (j == 99):
				result.write("\n")

	result.close()

if __name__ == "__main__":
	main()