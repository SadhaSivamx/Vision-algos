import numpy as np
class Disjointset:
    def __init__(self, n, m, k):
        self.k = k
        self.parent = np.indices((n, m)).transpose(1, 2, 0)
        self.size = np.ones((n, m))
        self.thresh = np.empty((n, m))
        self.thresh.fill(self.k)

    def findparent(self, u, v):
        if np.array_equal(self.parent[u, v], [u, v]):
            return (u, v)
        self.parent[u, v] = self.findparent(*self.parent[u, v])
        return tuple(self.parent[u, v].tolist())

    def merge(self, weight, pt1, pt2):
        u1, v1 = pt1
        u2, v2 = pt2

        par1 = self.findparent(u1, v1)
        par2 = self.findparent(u2, v2)

        if par1 == par2:
            return False

        if weight <= min(self.thresh[par1], self.thresh[par2]):
            if self.size[par1] >= self.size[par2]:
                self.parent[par2] = par1
                self.size[par1] += self.size[par2]
                self.thresh[par1] = weight + (self.k / self.size[par1])
            else:
                self.parent[par1] = par2
                self.size[par2] += self.size[par1]
                self.thresh[par2] = weight + (self.k / self.size[par2])
            return True
        return False
