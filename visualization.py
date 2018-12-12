import matplotlib.pyplot as plt


def plot_data_over_time(time, y, title=None, xlabel="Date", ylabel=None, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    fig.autofmt_xdate()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(time, y)
    plt.show()

def plot_fundamental_diagram(flow, occupancy, title=None):
    plt.title(title)
    plt.xlabel("Occupancy (%)")
    plt.ylabel("Flow (vph)")

    plt.scatter(occupancy, flow)
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.show()
