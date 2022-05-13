import sys
import data

def main(argv):
    df = data.load_data('/home/student/Homeworks//HW2//london_sample_500.csv')
    data.add_new_columns(df)
    print("Part A: ")
    data.data_analysis(df)


if __name__ == '__main__':
     main(sys.argv)