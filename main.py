import sys
import data
import clustering
import matplotlib.pyplot as plt


def main(argv):
    path = "./london_sample_500.csv"
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

    path_fig = "./plots.pdf"

    fig = plt.figure(figsize=(10, 12))

    ax1 = fig.add_axes([0.15, 0.7, 0.7, 0.22])
    clustering.visualize_results(num_data,labels_k2,centroids_k2, ax1)
    ax2 = fig.add_axes([0.15, 0.4, 0.7, 0.22])
    clustering.visualize_results(num_data,labels_k3,centroids_k3, ax2)
    ax3 = fig.add_axes([0.15, 0.1, 0.7, 0.22])
    clustering.visualize_results(num_data,labels_k5,centroids_k5, ax3)

    #plt.show()
    plt.savefig(path_fig)

if __name__ == '__main__':
     main(sys.argv)