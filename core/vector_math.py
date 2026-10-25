import numpy as np

class VectorMath:
    def __init__(self):
        self.matrix = None

    def add_vector(self, vector):
        if self.matrix is None:
            self.matrix = np.atleast_2d(vector)
        else:
            self.matrix = np.vstack([self.matrix, np.atleast_2d(vector)])

        return self.matrix.shape

    def search(self, query_vector, top_k=3):
        # In case the matrix is empty
        if self.matrix is None:
            return []
        
        q = np.array(query_vector)

        # Dot Product
        dot = np.dot(self.matrix, q)

        # L2 Norm
        norm = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(q)

        # Dviide dot by norm to get cosine similarity
        similarity = dot / norm
        similarity = np.argsort(similarity)[::-1][:top_k]

        return similarity.tolist()