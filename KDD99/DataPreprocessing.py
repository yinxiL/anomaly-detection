import argparse
from io import open
from Definition import *

parser = argparse.ArgumentParser(description='KDD99 Examples')
parser.add_argument('--dataset', type=str, default='10_percent', help='dataset to transform (10_percent/full)')
args = parser.parse_args()

def transform_type(input_file, output_file):
    with open(output_file, "w") as text_file:
        with open(input_file) as f:
            lines = f.readlines()
            for line in lines:
                columns = line.split(',')
                for raw_type in category:
                    flag = False
                    if raw_type == columns[-1].replace("\n", ""):
                        str = ','.join(columns[0:col_names.index('label')])
                        text_file.write("%s,%d\n" % (str, category[raw_type]))
                        flag = True
                        break
                if not flag:
                    text_file.write(line)
                    print(line)

if args.dataset == 'full':
	transform_type("data/kddcup.data.txt", "data/kddcup.data_transformed.txt")
else:
	transform_type("data/kddcup.data_10_percent.txt", "data/kddcup.data_10_percent_transformed.txt")
transform_type("data/corrected.txt", "data/corrected_transformed.txt")
