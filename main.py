import sys
import data
import clustering


def main(argv):
    path = "./london.csv"
    df = data.load_data(path)
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

    clustering.visualize_results(num_data, labels_k2, centroids_k2, "./plt2.png")
    clustering.visualize_results(num_data, labels_k3, centroids_k3, "./plt3.png")
    clustering.visualize_results(num_data, labels_k5, centroids_k5, "./plt5.png")

if __name__ == '__main__':
     main(sys.argv)