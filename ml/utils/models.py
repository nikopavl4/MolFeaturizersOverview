import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import tempfile

def create_model(tasks,args):
    model_dir = tempfile.mkdtemp()
    if (args.model.lower() == 'svm'):
        model = dc.models.SingletaskToMultitask(tasks, svm_model_builder, model_dir)
    elif (args.model.lower() == 'randomforest'):
        model = dc.models.SingletaskToMultitask(tasks, random_forest_builder, model_dir)
    elif (args.model.lower() == 'gradboost'):
        model = dc.models.SingletaskToMultitask(tasks, gradboost_builder, model_dir)
    elif (args.model.lower() == 'convmol'):
        model = dc.models.GraphConvModel(len(tasks), batch_size=50, mode='classification')
    elif (args.model.lower() == 'weave'):
        model = dc.models.WeaveModel(n_tasks=len(tasks), n_weave=2, fully_connected_layer_sizes=[200, 100], mode="classification")
    else:
        print("No such model")

    return model


def svm_model_builder(model_dir):
    sklearn_model = SVC(C=1.0, class_weight="balanced", probability=True)
    return dc.models.SklearnModel(sklearn_model, model_dir)

def random_forest_builder(model_dir):
    sklearn_model = RandomForestClassifier()
    return dc.models.SklearnModel(sklearn_model, model_dir)

def gradboost_builder(model_dir):
    sklearn_model = GradientBoostingClassifier()
    return dc.models.SklearnModel(sklearn_model, model_dir)

