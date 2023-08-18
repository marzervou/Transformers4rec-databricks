
import fasttext
import mlflow.pyfunc


class MerlinRecWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to train and use merlin recommender Models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """
        import fasttext

        self.model = fasttext.load_model(context.artifacts["merlin_model_path"])

    def predict(self, context, model_input):
        """This is an abstract function. We customized it into a method to fetch the merlin model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.
        Returns:
            [type]: the loaded model artifact.
        """
        return self.model