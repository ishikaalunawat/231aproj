import matplotlib.pyplot as plt

def read_log_file(file_name):

    loss_values = []

    with open(file_name, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Epoch:"):
            loss_values.append(float(line.split(" ")[-1]))

    fig, ax = plt.subplots()
    ax.plot(range(int(len(loss_values)/2)), loss_values[::2], marker='o')

    ax.set_title('Training Loss over epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('L2 Loss Value')
    ax.set_ylim(0, 0.01)

    plt.show()

if __name__=="__main__":
    read_log_file("logfile.log")