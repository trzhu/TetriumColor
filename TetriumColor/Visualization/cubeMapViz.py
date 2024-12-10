import matplotlib.pyplot as plt


def SetUp3DPlot(limits=(-0.5, 0.5)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.axis('equal')
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[0], limits[1])
    ax.set_zlim(limits[0], limits[1])

    return ax
