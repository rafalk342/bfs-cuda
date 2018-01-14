import sys
from random import choice

if __name__ == "__main__":
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    if m < n - 1:
        print("Cannot create connected graph.")
        exit(1)

    print(n, m)

    edges = set()
    not_used = [i for i in range(1, n)]
    used = [0]

    while m > 0:
        if len(not_used):
            u = choice(not_used)
            v = choice(used)
            print(u, v)
            edges.add((u, v))
            edges.add((v, u))
            not_used.remove(u)
            used.append(u)
            m -= 1
        else:
            u = choice(used)
            v = choice(used)
            if u != v and not (u, v) in edges:
                print(u, v)
                edges.add((u, v))
                edges.add((v, u))
                m -= 1
