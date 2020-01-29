import matplotlib.pyplot as plt


def graph_plot():
    P = ([1, 0], [1, 1], [1, 1], [1, 1], [1, 1])
    Q = ([0, 1], [1, 0], [1, 1], [1, 1], [2, 2])
    origin = ([0, 0])
    p_x = []
    p_y = []
    q_x = []
    q_y = []
    for p, q in zip(P, Q):
        p_x.append(p[0])
        p_y.append(p[1])
        q_x.append(q[0])
        q_y.append(q[1])
    print(p_x, p_y, q_x, q_y)
    # for i, j in zip(P, Q):
    #     plt.plot(i, j)
    #     plt.show()
    #     plt.pause(5)


if __name__ == "__main__":
    graph_plot()
