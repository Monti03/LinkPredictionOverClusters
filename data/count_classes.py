
import sys

from collections import Counter

if __name__ == "__main__":
    path = str(sys.argv[1])
    labels = []
    with open(path) as fin:
        for line in fin:
            labels.append(line.strip())
    
    d = Counter(labels)

    print("n_nodes:", len(labels))
    for k in d:
        print(f"label {k}: ", d[k])