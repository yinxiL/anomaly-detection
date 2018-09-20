from io import open
from Definition import *

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

transform_type("data/kddcup.data_10_percent.txt", "data/kddcup.data_10_percent_transformed.txt")
transform_type("data/corrected.txt", "data/corrected_transformed.txt")
