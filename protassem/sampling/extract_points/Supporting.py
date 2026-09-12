import numpy as np

from protassem.core.points_txt import read_point_cloud

# Create Directory
def MkdirList(folderlist):
    import os
    for i in folderlist:
        if not os.path.exists('%s' % (i)):
            os.mkdir('%s' % (i))


# Write a pymol readable XYZ
def XYZ(listofarray, label, fn):
    XYZTrial = []
    if listofarray:
        Points = listofarray
        for point in Points:
            XYZTrial.append('%.5f        %.5f        %.5f\n' % (point[0], point[1], point[2]))
        with open("%s" % (fn), 'w') as f:
            for point in XYZTrial:
                f.write(point)
    del XYZTrial


def XYZ_TXT(listofarray,fn):
    XYZTrial = []
    if listofarray:
        Points=listofarray
        for point in Points:
            XYZTrial.append('%.5f        %.5f        %.5f\n' %(point[0], point[1], point[2]))
        with open("%s" %(fn),'a') as f:
            for point in XYZTrial:
                f.write(point)
    del XYZTrial


# Chunk
def chunks(l, n):
    # list(chunks(range(10, 75), 10))
    for i in range(0, len(l), n):
        yield l[i:i + n]


def Misc_Centroid_Calculation(i, Y, labels):
    return np.mean(Y[labels == i], axis=0)


def filter_cluster_noise(X, labels):
    indices = np.where(labels >= 0)
    return X[indices, :], labels[indices, :]


def cluster_mean_distance(X, labels):

    X, labels = filter_cluster_noise(X, labels)
    label_freqs = np.bincount(labels)
    clust_dists = np.zeros((len(X), len(label_freqs)), dtype=X.dtype)
    for i in range(len(X)):
        clust_dists[i] += np.bincount(
            labels, weights=X[i], minlength=len(label_freqs)
        )

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(X)), labels)
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    return intra_clust_dists


def load_sample_points(file_path, density=False):
    """读取点云 TXT；格式契约与解析规则见 protassem.core.points_txt。"""
    cloud = read_point_cloud(file_path)
    if density:
        return cloud.points, cloud.vectors, cloud.densities.reshape(-1, 1)
    return cloud.points, cloud.vectors


def txt2pcd(pcd_points, output):

    with open(output, 'w') as f:
        f.writelines("# .PCD v0.7 - Point Cloud Data file format\n")
        f.writelines("VERSION 0.7\n")
        f.writelines("FIELDS x y z\n")
        f.writelines("SIZE 4 4 4\n")
        f.writelines("TYPE F F F\n")
        f.writelines("COUNT 1 1 1\n")
        f.writelines("WIDTH " + str(pcd_points.shape[0]) + "\n")
        f.writelines("HEIGHT 1\n")
        f.writelines("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.writelines("POINTS " + str(pcd_points.shape[0]) + "\n")
        f.writelines("DATA ascii\n")

        '''for i in range(len(xlist)):
            file_to_write.writelines(str(xlist[i]) + " " + str(ylist[i]) + " " + str(zlist[i]) + "\n")'''

        for i in range(pcd_points.shape[0]):
            f.write(" ".join(["%.5f" % j for j in pcd_points[i]]))
            f.write("\n")

