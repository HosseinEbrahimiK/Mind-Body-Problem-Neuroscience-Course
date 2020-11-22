import os
import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt


class NeuronProfile:
    def __init__(self, name):
        self.responses = None
        self.name = name
        self.responses = []
        self.spike_count_rate = 0


# part 2.4
def func_stimuli_extraction(events, mat, f):

    output_matrix = []
    for spike in events:

        num_of_frame = int(np.floor(f * (spike / 10000)))

        if num_of_frame < 15:
            m = np.pad(mat[:num_of_frame+1], ((0, 15-num_of_frame), (0, 0)), mode='constant')
            output_matrix.append(np.array(m))

        else:
            if num_of_frame == len(mat):
                output_matrix.append(np.array(mat[-16:]))
            else:
                output_matrix.append(np.array(mat[num_of_frame-15:num_of_frame+1]))

    return output_matrix


# part 3.1
def get_spike_trigger_average(neuro, mat, f):

    spike_trigger_average = []
    for i in range(len(neuron.responses)):
        spike_trigger_mat = func_stimuli_extraction(neuro.responses[i], mat, f)
        spike_trigger_average.append(np.mean(spike_trigger_mat, axis=0))

    out = np.mean(spike_trigger_average, axis=0)

    return out


# part 3.2
def calculate_t_test(neuro, mat, f):

    all_spikes = list()
    for k in range(len(neuro.responses)):
        all_spikes.append(np.array(func_stimuli_extraction(neuro.responses[k], mat, f)))

    all_spikes = np.array(all_spikes)

    values = [[[] for _ in range(16)] for _ in range(16)]
    p_values = [[0 for _ in range(16)] for _ in range(16)]

    for i in range(len(all_spikes)):
        for j in range(len(all_spikes[i])):
            for k in range(16):
                for r in range(16):
                    values[k][r].append(all_spikes[i][j][k][r])

    for i in range(16):
        for j in range(16):
            p_values[i][j] = stats.ttest_1samp(values[i][j], 0)[1]

    return p_values


# part 3.3
def projection_on_spike_trigger_average(neuro, mat, f):

    all_spikes = list()
    for k in range(len(neuro.responses)):
        all_spikes.append(np.array(func_stimuli_extraction(neuro.responses[k], mat, f)))

    vectors = list()
    for i in range(len(all_spikes)):
        for j in range(len(all_spikes[i])):
            vectors.append(np.array(all_spikes[i][j]).reshape(256,))

    average_vector = np.array(get_spike_trigger_average(neuro, mat, f)).reshape(256,)
    projections = list()

    control_projection = list()

    for i in range(len(vectors)):
        random_index = np.random.randint(15, len(mat)-1)
        m = np.array(mat[random_index-15:random_index+1]).reshape(256,)

        projections.append((np.dot(vectors[i], average_vector) / np.linalg.norm(average_vector)))
        control_projection.append((np.dot(m, average_vector) / np.linalg.norm(average_vector)))

    return projections, control_projection


# part 4.1
def spike_triggered_correlation(event, mat, f):

    spike_stimuli = np.array(func_stimuli_extraction(event, mat, f))
    spikes = list()
    for spike in spike_stimuli:
        spikes.append(spike.reshape(256,))

    spikes = np.array(spikes)

    correlation_matrix = [[0 for _ in range(256)] for _ in range(256)]

    for i in range(256):
        for j in range(256):
            cnt = 0
            for k in range(len(spikes)):
                cnt += (spikes[k][i] * spikes[k][j])

            correlation_matrix[i][j] = cnt / len(spikes)

    return correlation_matrix


# part 4.3
def projection_on_eig_vector(event, mat, f):

    corr_matrix = spike_triggered_correlation(event, mat, f)
    val, vectors = np.linalg.eig(np.array(corr_matrix))

    arr = [[k, val[k]] for k in range(len(val))]
    arr.sort(key=lambda item: item[1], reverse=True)

    big_vectors = [vectors[arr[k][0]] for k in range(2)]

    spikes = func_stimuli_extraction(event, mat, f)
    projection0 = list()
    projection1 = list()

    vectors_of_spikes = list()
    for i in range(len(spikes)):
        vectors_of_spikes.append(np.array(spikes[i]).reshape(256,))

    control0, control1 = list(), list()

    for i in range(len(vectors_of_spikes)):

        random_index0 = np.random.randint(15, len(mat) - 1)
        random_index1 = np.random.randint(15, len(mat) - 1)
        m0 = np.array(mat[random_index0 - 15:random_index0 + 1]).reshape(256, )
        m1 = np.array(mat[random_index1 - 15:random_index1 + 1]).reshape(256, )

        projection0.append((np.dot(big_vectors[0], vectors_of_spikes[i])) / np.linalg.norm(big_vectors[0]))
        projection1.append((np.dot(big_vectors[1], vectors_of_spikes[i])) / np.linalg.norm(big_vectors[1]))

        control0.append((np.dot(big_vectors[0], m0)) / np.linalg.norm(big_vectors[0]))
        control1.append((np.dot(big_vectors[1], m1)) / np.linalg.norm(big_vectors[1]))

    return projection0, control0, projection1, control1


if __name__ == "__main__":

    # loading data from .csv files that are beside of this file

    directory = os.listdir("data")
    neurons = []
    names = []
    for i in range(len(directory)):

        if directory[i][0] == '0':
            string = directory[i][:6]

            if string not in names:

                neurons.append(NeuronProfile(string))

                for j in range(len(directory)):

                    if directory[j][:6] == string and directory[j][-3:] == "csv":
                        # ******************** CHANGE ADDRESS OF DATA TO RUN CODE ***********************
                        neurons[-1].responses.append(np.array(np.genfromtxt("data/"+directory[j], delimiter=',')))

                names.append(string)

    # calculating spike-rate-count of neurons, part 2.3

    matrix = np.array(sio.loadmat("msq1D.mat")['msq1D'])
    freq = 59.721395
    time_of_experiment = len(matrix) * (1 / freq)

    for single_neuron in neurons:
        count = 0
        for r in single_neuron.responses:
            count += len(r)

        single_neuron.spike_count_rate = count / (len(single_neuron.responses) * time_of_experiment)

    count_rates = []
    print("Neurons with less than 2 spike-count-rate:")
    for neuron in neurons:
        if neuron.spike_count_rate < 2:
            print(neuron.name)
        count_rates.append(neuron.spike_count_rate)

    plt.bar(names, count_rates, color='blue')
    plt.title("Histogram of Spike-Count-Rate")
    plt.show()

    # for single neuron, part 3
    print("code of neuron: ", neurons[1].name)
    print("event: ", neurons[1].responses[0])

    # showing RF of neuron

    img_mat = get_spike_trigger_average(neurons[1], matrix, freq)
    plt.imshow(img_mat, cmap='gray')
    plt.title("Receptive Field of Mentioned Neuron ")
    plt.show()

    # p-value of mentioned neuron

    p = calculate_t_test(neurons[1], matrix, freq)
    plt.imshow(p, cmap='gray')
    plt.title("P-value: ")
    plt.show()

    projection_list, random_list = projection_on_spike_trigger_average(neurons[1], matrix, freq)
    plt.hist(projection_list, density=True, bins=25, color='blue')
    plt.hist(random_list, density=True, bins=25, color='orange')
    plt.title("Spike Trigger Projection and Control Projection")
    plt.show()

    # part 3.4
    t_test, p_val = stats.ttest_ind(projection_list, random_list)
    print("spike trigger average: ")
    print("     t_test_score : ", t_test)
    print("     p-value : ", p_val)

    # for all neurons, part 3.5

    # for neuron in neurons:
    #   show_img = get_spike_trigger_average(neuron, matrix, freq)
    #    plt.imshow(show_img, cmap='gray')
    #    plt.show()

    #    prob = calculate_t_test(neuron, matrix, freq)
    #    plt.imshow(prob, cmap='gray')
    #    plt.show()

    #    spike, control = projection_on_spike_trigger_average(neuron, matrix, freq)
    #    plt.hist(spike, normed=True, bins=25, color='blue')
    #    plt.hist(control, normed=True, bins=25, color='orange')
    #    plt.show()

    # part 4.1 getting correlation matrix and calculating eign values and eign vectors

    corr_mat = np.array(spike_triggered_correlation(neurons[1].responses[0], matrix, freq))
    eig_Value, eig_Vector = np.linalg.eig(corr_mat)

    lst = [[k, eig_Value[k]] for k in range(len(eig_Value))]
    lst.sort(key=lambda item: item[1], reverse=True)

    biggest_eigVector = [eig_Vector[lst[k][0]] for k in range(3)]

    for i in range(len(biggest_eigVector)):
        biggest_eigVector[i] = biggest_eigVector[i].reshape(16, 16)
        plt.imshow(biggest_eigVector[i], cmap='gray')
        plt.title("Projection on No. Biggest eign vector")
        plt.show()

    # projection on two eign vector corresponding to biggest eign values
    project_eig0, ctrl_eig0, project_eig1, ctrl_eig1 = projection_on_eig_vector(neurons[1].responses[0], matrix, freq)

    plt.hist(project_eig0, density=True, bins=25, color='blue')
    plt.hist(ctrl_eig0, density=True, bins=25, color='orange')
    plt.title("part : 4.3")
    plt.show()

    plt.hist(project_eig1, density=True, bins=25, color='blue')
    plt.hist(ctrl_eig1, density=True, bins=25, color='orange')
    plt.title("part :4.3")
    plt.show()
