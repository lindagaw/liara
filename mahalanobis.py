import numpy as np

def emp_mean(set):
    norms = []
    for item in set:
        norm = np.linalg.norm(item)
        norms.append(norm)

    return np.mean(norm)

def emp_covar(set, emp_mean):
    vals = []
    for item in set:
        norm = np.linalg.norm(item)
        val = (norm - emp_mean) * (norm - emp_mean)
        vals.append(val)
    return np.mean(vals)

def mahalanobis_loss(set, image):


    miu = emp_mean(set)
    sigma = emp_covar(set, miu)
    norm = np.linalg.norm(image.detach().numpy())
    m = (norm - miu) * sigma * (norm - miu) / 100000000

    return m
