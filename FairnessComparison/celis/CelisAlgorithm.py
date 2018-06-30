from algorithms.Algorithm import Algorithm
import numpy
import tempfile
import os
import subprocess
import json
import sys
import numpy

class CelisAlgorithmBase(Algorithm):

    def __init__(self):
        Algorithm.__init__(self)

    def get_supported_data_types(self):
        return set(["numerical-binsensitive"])

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):

        value_0 = train_df[class_attr].values[0]
        if type(value_0) == str:
            class_type = str
        else:
            class_type = type(value_0.item()) # this should be numpy.int64 or numpy.int32,

        def create_file(df):
            out = {}
            out["x"] = df.drop(columns=[class_attr]).as_matrix().tolist()
            out["class"] = (2 * df[class_attr] - 1).as_matrix().tolist()
            out["sensitive"] = {}
            out["sensitive"][single_sensitive] = df[single_sensitive].as_matrix().tolist()
            fd, name = tempfile.mkstemp()
            os.close(fd)
            out_file = open(name, "w")
            json.dump(out, out_file)
            out_file.close()
            return name
        
        train_name = create_file(train_df)
        test_name = create_file(test_df)
        fd, predictions_name = tempfile.mkstemp()
        os.close(fd)
        # print("CURRENT DIR: %s" % os.getcwd())
        # print("SENSITIVE ATTR: %s" % single_sensitive)

        cmd = self.create_command_line(train_name, test_name, predictions_name, params)
        result = subprocess.run(cmd,
            cwd='./algorithms/celis/')
        os.unlink(train_name)
        os.unlink(test_name)
        if result.returncode != 0:
            os.unlink(predictions_name)
            raise Exception("Algorithm did not execute succesfully")
        else:
            predictions = open(predictions_name).read()
            predictions = json.loads(predictions)
            os.unlink(predictions_name)
            # m = numpy.loadtxt(output_name)
            # os.unlink(output_name)

            # predictions = m[:,1]
            predictions_correct = [0 if class_type(x) == -1 else 1 for x in predictions]

            # print("Predictions:  %s" % predictions_correct)
            # print("ground truth: %s" % test_df[class_attr].as_matrix().tolist())
            return predictions_correct, []

##############################################################################


class CelisAlgorithmFairDiscovery(CelisAlgorithmBase):

    def __init__(self):
        CelisAlgorithmBase.__init__(self)
        self.name = "CelisFairDiscovery"

    def get_param_info(self):
        return {'tau': np.linspace(0.1,1,10)}

    def get_default_params(self):
        return {'tau': 0.8}

    def create_command_line(self, train_name, test_name, predictions_name, params):
        return ['python3', 'mainFDR.py',
                train_name,
                test_name,
                predictions_name, str(params['tau'])]

class CelisAlgorithmStatParity(CelisAlgorithmBase):

    def __init__(self):
        CelisAlgorithmBase.__init__(self)
        self.name = "CelisStatParity"

    def get_param_info(self):
        return {'tau': np.linspace(0.1,1,10)}

    def get_default_params(self):
        return {'tau': 0.8}

    def create_command_line(self, train_name, test_name, predictions_name, params):
        return ['python3', 'mainSR.py',
                train_name,
                test_name,
                predictions_name, str(params['tau'])]

