import sys
import data
import clustering

def main(argv):
    df = data.load_data('/home/student/Homeworks//HW2//london_sample_500.csv')
    data.add_new_columns(df)
    print("Part A: ")
    data.data_analysis(df)

    print()
    print("Part B: ")
    features = ["cnt", "hum"]
    num_data = clustering.transform_data(df, features)
    labels_k2, centroids_k2 = clustering.kmeans(num_data,2)
    labels_k3, centroids_k3 = clustering.kmeans(num_data,3)
    labels_k5, centroids_k5 = clustering.kmeans(num_data,5)

    print("k = 2")
    print(centroids_k2)
    print()
    print("k = 3")
    print(centroids_k3)
    print()
    print("k = 5")
    print(centroids_k5)

    path = '//home//student//Homeworks//HW2'
 #   clustering.visualize_results(num_data,labels,centroids, path)


if __name__ == '__main__':
     main(sys.argv)