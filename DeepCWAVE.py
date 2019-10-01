import numpy as np
import itertools
import argparse
import pickle

class DCWAVELoader:
    def __init__(self, weight_file='./model/hsALT_regressor.pkl'):
        self.MDL_PARAMS = weight_file
        self.NUM_LAYERS = 0
        self.WEIGHTS = []
        self.BIASES = []
        self._load_params()
        # maps standardization constants to appropriate columns by index
        self.PARAMS = {'3_0': {'mu': 23.67335083234615, 'sig': 0.5304684274016728},  # incidenceAngle mode 0
                       '3_1': {'mu': 122.71640428451626, 'sig': 2371.766614973607},  # incidenceAngle mode 1
                       '4': {'mu': 6.483141259641842e+33, 'sig': 2.5414516590226384e+35},  # sigma naught
                       '5': {'mu': 6.483141259641842e+33, 'sig': 2.5414516590226384e+35},  # normalized Variance
                       '6': {'mu': 8.841854523752763, 'sig': 3.1408964129681483},  # s0
                       '7': {'mu': 0.961449130322787, 'sig': 5.947997039490825},  # s1
                       '8': {'mu': 2.0505311806163866, 'sig': 2.9171238725610356},  # s2
                       '9': {'mu': 0.09299508623428353, 'sig': 3.2646609635356527},  # s3
                       '10': {'mu': -6.008289862068798, 'sig': 3.267945544630491},  # s4
                       '11': {'mu': 2.5545143231216767, 'sig': 2.2710846538656875},  # s5
                       '12': {'mu': -0.5556251936821094, 'sig': 2.6549735757402417},  # s6
                       '13': {'mu': 2.091213170844702, 'sig': 1.74363806951232},  # s7
                       '14': {'mu': -0.08347184538552771, 'sig': 1.9326049771111427},  # s8
                       '15': {'mu': 0.8833822560820885, 'sig': 2.432405795063978},  # s9
                       '16': {'mu': -1.4491221214924368, 'sig': 2.303733312526022},  # s10
                       '17': {'mu': -0.6760330906344267, 'sig': 2.8772594446556634},  # s11
                       '18': {'mu': 1.0409890935835604, 'sig': 1.394046234693587},  # s12
                       '19': {'mu': -0.2025180155930731, 'sig': 1.7721983460630586},  # s13
                       '20': {'mu': 2.760592845356428, 'sig': 1.7501487173110526},  # s14
                       '21': {'mu': -0.0030536411756998225, 'sig': 2.9695940008831947},  # s15
                       '22': {'mu': -0.7782292745154133, 'sig': 3.714414899604148},  # s16
                       '23': {'mu': 1.4197555343821817, 'sig': 1.1947407017447227},  # s17
                       '24': {'mu': -0.18107631076928576, 'sig': 1.4009414430164768},  # s18
                       '25': {'mu': 2.9972522736604064, 'sig': 3.4168292647748664}}  # s19

    @staticmethod
    def _relu(x):
        """Relu activation function for NN

        Args:
            x: data matrix

        Returns:
            relu(x)
        """
        np.maximum(x, 0, x)
        return x

    @staticmethod
    def _il(x):
        """Approaches 0 for x-> -inf as 1/(1-x), with f(0)=1 and f'(0)=1. Cont differentiable.
        Uses TF workaround from: https://github.com/tensorflow/tensorflow/issues/2540

        Args:
            x: data matrix

        Returns:
            relu(x)
        """
        if x < 0:
            return 1 / (1 - x)
        else:
            return x + 1

    @staticmethod
    def _standardize(data, params):
        """Standardizes data matrix to mean=0, std=1

        Args:
            data: data matrix
            params: dict with pre-calculate mu and sig values for data

        Returns:
            data-mean/std
        """
        m = params['mu']
        s = params['sig']
        return (data - m) / s

    @staticmethod
    def _conv_time(in_t):
        """Converts data acquisition time

        Args:
            in_t: time of data acquisition in format hours since 2010-01-01T00:00:00Z UTC

        Returns:
            Encoding of time where 00:00 and 24:00 are -1 and 12:00 is 1
        """
        in_t = in_t % 24
        return 2 * np.sin((2 * np.pi * in_t) / 48) - 1

    @staticmethod
    def _conv_deg(in_angle, is_inverse=False, in_cos=None, in_sin=None):
        """Converts measurements in degrees (e.g. angles), using encoding proposed at https://stats.stackexchange.com/a/218547
           Encode each angle theta as tuple (cos(theta), sin(theta))

        Args:
            coord: measurement of lat/ long in degrees

        Returns:
            tuple of values between -1 and 1
        """
        if is_inverse:
            return np.sign(np.rad2deg(np.arcsin(in_sin))) * np.rad2deg(np.arccos(in_cos))

        angle = np.deg2rad(in_angle)
        return [np.cos(angle), np.sin(angle)]

    def _conv_incAng(self, in_angle):
        """Converts incidence angle into tuple of values, one angle and one a binary label for wave mode
           Label is 0 if angle is around 23 and 1 if angle is around 37

        Args:
            in_angle: angle, either around 23 or 37 degrees

        Returns:
            tuple of values, one just the angle, the other a binary label
        """
        ang_type = int(in_angle > 30)
        if ang_type == 0:
            in_angle = self._standardize(in_angle, self.PARAMS['3_0'])
        else:
            in_angle = self._standardize(in_angle, self.PARAMS['3_1'])
        return [in_angle, ang_type]

    def _proc_data(self, data_mat):
        """Preprocessing step. Applies appropriate normalization techniques based on column index.

        Args:
            data_mat: data matrix

        Returns:
            preprocessed data matrix
        """
        data = np.array(data_mat, dtype='float32')
        for idx, col in enumerate(data.T):
            if idx == 0:  # time of day SAR
                form_data = np.array(list(map(self._conv_time, col)),dtype='float32')
            elif idx == 1 or idx == 2:  # longitude& latitude
                form_col = np.array(list(map(self._conv_deg, col)),dtype='float32')
                form_data = np.vstack([form_data, np.array(form_col[:,0], dtype='float32')])
                form_data = np.vstack([form_data, np.array(form_col[:,1], dtype='float32')])
            elif idx == 3:  # incidence Angle
                # add dx and dt columns:
                form_data = np.vstack([form_data, np.zeros((2, col.shape[0]))])
                # get and add normalized incidence angle and binary indicator variable
                form_col = np.array(list(map(self._conv_incAng, col)), dtype='float32')
                form_data = np.vstack([form_data, np.array(form_col[:,0], dtype='float32')])
                form_data = np.vstack([form_data, np.array(form_col[:,1], dtype='float32')])
            elif 2<idx<26:  # orthogonal parameters
                # sorry:
                form_col = np.array(list(map(self._standardize, col, itertools.repeat(self.PARAMS[str(idx)], col.shape[0]))),
                                    dtype='float32')
                form_data = np.vstack([form_data, form_col])
            else:  # sentinel type
                form_data = np.vstack([form_data, np.int8(col)])
        return form_data.T

    def _load_params(self):
        """Loads neural network weights and biases from pickle file

        """
        with open(self.MDL_PARAMS, 'rb') as handle:
            params = pickle.load(handle)
            for key in list(params.keys()):
                layer = params[key]
                l_weights = layer['weights']
                l_biases = layer['biases']
                self.WEIGHTS.append(l_weights)
                self.BIASES.append(l_biases)
                self.NUM_LAYERS += 1

    def _layer_output(self, inp, weights, biases, is_out=False):
        """Calculates output for standard dense feed-forward layer.

        Args:
            inp: input data
            weights: layer weights
            biases: layer biases
            is_out: boolean, changes activation function if true

        Returns:
            model output
        """
        if is_out:
            out = np.dot(inp, weights) + biases
        else:
            out = np.dot(inp, weights) + biases
            out = self._relu(out)
        return out

    def _final_output(self, inp):
        """Transforms linear output from network to parameterize heteroskedastic Gaussian distribution

        Args:
            inp: model output

        Returns:
            transformed model output
        """
        mu_out = inp[:, 0]
        sig_out = inp[:, 1]
        sig_out = np.array(list(map(self._il, sig_out)), dtype='float32')
        out = np.stack((mu_out, sig_out), axis=1)
        return out

    def predict(self, x):
        """Driver function, preprocesses raw input and returns model predictions

        Args:
            x: data matrix, has to be of shape (n,27)

        Returns:
            model outputs of shape (n,2) with first output dimension being mean and second being std for Gaussian
        """
        x = self._proc_data(x)
        for l_idx in range(self.NUM_LAYERS):
            # intermediate hidden layers
            if l_idx < self.NUM_LAYERS - 1:
                x = self._layer_output(x, self.WEIGHTS[l_idx], self.BIASES[l_idx], False)
            # final hidden layer
            else:
                x = self._layer_output(x, self.WEIGHTS[l_idx], self.BIASES[l_idx], True)
        return self._final_output(x)


if __name__ =='__main__':
    #If ran as standalone script
    parser = argparse.ArgumentParser(description='DeepCWAVE model')

    parser.add_argument('input', nargs='+', type=float,
                        help='Space delimited input to CWAVE')

    parser.add_argument(
        '--weights', '-w', type=str, required=False,
        help='File containing model parameters. Defaults to hsALT_regressor.pkl file in ./models/ directory.'
    )

    usr_args = vars(parser.parse_args())

    if usr_args['weights']:
        mdl = DCWAVELoader(weight_file=usr_args['weights'])
    else:
        mdl = DCWAVELoader()

    inp = np.array(usr_args['input']).reshape(1,27)
    print(mdl.predict(inp))

else:
    # if imported as module
    mdl = DCWAVELoader()
    def predict(x, weight_file=None):
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype='float32')
        if np.ndim(x)<2:
            x = x.reshape(1,27)
        assert x.shape[1] == 27

        if weight_file:
            tmp_mdl = DCWAVELoader(weight_file=weight_file)
            preds = tmp_mdl.predict(x)
        else:
            preds = mdl.predict(x)
        return preds
