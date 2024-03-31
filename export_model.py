from network import create_fc_network
from keras_tools.export import export_current_model
import keras.backend as K

#subjects = range(10, 51)
subjects = [41]
nr_of_objects = 6
experiment_name = 'flip_gamma_rotate_shadow_elastic'
for subjectID in subjects:
    if subjectID == 9 or subjectID == 43:
        continue

    model = create_fc_network(nr_of_objects, input_shape=(256, 256, 1))
    model.load_weights('models/model_' + str(subjectID) + '_' + experiment_name + '.hdf5')

    print('Inputs: ', model.input)
    print('Outputs: ', model.output)

    output_name = model.output.name.split(':')[0]

    export_current_model('models/axillary_block_' + str(subjectID) + '_' + experiment_name + '.pb', output_name)
    K.clear_session()

