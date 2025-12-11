from abc import ABC, abstractmethod
from enum import Enum

from darts.utils.likelihood_models import QuantileRegression
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    AutoARIMA,
    TiDEModel as DartsTiDEModel,
    NBEATSModel as DartsNBEATSModel,
    NHiTSModel as DartsNHiTSModel,
    XGBModel as DartsXGBModel,
    Chronos2Model as DartsChronos2Model,
)

quantiles = [
    0.01,
    # 0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    # 0.975,
    0.99,
]


class TrainingWindowType(Enum):
    EXPANDING = "expanding"
    ROLLING = "rolling"


class Model(ABC):
    def __init__(self, *args, **kwargs):
        self.params_args = args
        self.params_kwargs = kwargs
        self.model = self.create_model(*self.params_args, **self.params_kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    @abstractmethod
    def reform(self, **kwargs):
        raise NotImplementedError("reform must be implemented by subclasses")

    @abstractmethod
    def create_model(self, *args, **kwargs):
        raise NotImplementedError("create_model must be implemented by subclasses")


class ModelRollingWindow(Model):
    def reform(self, **kwargs):
        # By default models are not reformed at each prediction step, instead they
        # use the base model. However, this behaviour can be overriden.
        pass


class ModelExpandingWindow(Model):
    def reform(self, **kwargs):
        # Form parameters with new kwargs overrides
        kwargs = {**self.params_kwargs, **kwargs}
        self.model = self.create_model(*self.params_args, **kwargs)


class NHiTSModelRollingWindow(ModelRollingWindow):
    def create_model(
        self, input_chunk_length, output_chunk_length, likelihood, output_chunk_shift=0
    ):
        return DartsNHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            # output_chunk_shift=output_chunk_shift,
            likelihood=likelihood,
        )


class NHiTSModelExpandingWindow(ModelExpandingWindow):
    def create_model(
        self, input_chunk_length, output_chunk_length, likelihood, output_chunk_shift=0
    ):
        return DartsNHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            # output_chunk_shift=output_chunk_shift,
            likelihood=likelihood,
        )


class NBEATSModelRollingWindow(ModelRollingWindow):
    def create_model(self, input_chunk_length, output_chunk_length, likelihood):
        return DartsNBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=likelihood,
        )


class NBEATSModelExpandingWindow(ModelExpandingWindow):
    def create_model(self, input_chunk_length, output_chunk_length, likelihood):
        return DartsNBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=likelihood,
        )


class TiDEModelRollingWindow(ModelRollingWindow):
    def create_model(self, input_chunk_length, output_chunk_length, likelihood):
        return DartsTiDEModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=likelihood,
        )


class TiDEModelExpandingWindow(ModelExpandingWindow):
    def create_model(self, input_chunk_length, output_chunk_length, likelihood):
        return DartsTiDEModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=likelihood,
        )


class XGBoostModelRollingWindow(ModelRollingWindow):
    def create_model(self, lags, output_chunk_length):
        return DartsXGBModel(
            lags=lags,
            output_chunk_length=output_chunk_length,
            likelihood="quantile",
            quantiles=quantiles,
            lags_past_covariates=lags,
        )


class XGBoostModelExpandingWindow(ModelExpandingWindow):
    def create_model(self, lags, output_chunk_length):
        return DartsXGBModel(
            lags=lags,
            output_chunk_length=output_chunk_length,
            likelihood="quantile",
            quantiles=quantiles,
            lags_past_covariates=lags,
        )


class Chronos2ModelRollingWindow(ModelRollingWindow):
    def create_model(self, input_chunk_length, output_chunk_length, likelihood):
        return DartsChronos2Model(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=likelihood,
        )


class Chronos2ModelExpandingWindow(ModelExpandingWindow):
    def create_model(self, input_chunk_length, output_chunk_length, likelihood):
        return DartsChronos2Model(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=likelihood,
        )


def get_model(model_name, training_window_type, max_horizon):
    match model_name:
        case "drift":
            label = "Drift"
            model = NaiveDrift()
        case "seasonal":
            label = "Seasonal"
            model = NaiveSeasonal(K=12)
        case ("arima", "autoarima", "sarima"):
            label = "AutoARIMA"
            model = AutoARIMA()
        case "tide":
            label = "TiDE"
            if training_window_type == TrainingWindowType.ROLLING:
                model_class = TiDEModelRollingWindow
            elif training_window_type == TrainingWindowType.EXPANDING:
                model_class = TiDEModelExpandingWindow
            else:
                raise ValueError(
                    f"Training window type {training_window_type} not recognized"
                )
            model = model_class(
                input_chunk_length=24,
                output_chunk_length=max_horizon,
                likelihood=QuantileRegression(quantiles=quantiles),
            )
        case "n-hits" | "nhits":
            label = "N-HiTS"
            if training_window_type == TrainingWindowType.ROLLING:
                model_class = NHiTSModelRollingWindow
            elif training_window_type == TrainingWindowType.EXPANDING:
                model_class = NHiTSModelExpandingWindow
            else:
                raise ValueError(
                    f"Training window type {training_window_type} not recognized"
                )
            model = model_class(
                input_chunk_length=24,
                output_chunk_length=max_horizon,
                likelihood=QuantileRegression(quantiles=quantiles),
            )
        case "n-beats" | "nbeats":
            label = "N-BEATS"
            if training_window_type == TrainingWindowType.ROLLING:
                model_class = NBEATSModelRollingWindow
            elif training_window_type == TrainingWindowType.EXPANDING:
                model_class = NBEATSModelExpandingWindow
            else:
                raise ValueError(
                    f"Training window type {training_window_type} not recognized"
                )
            model = model_class(
                input_chunk_length=24,
                output_chunk_length=max_horizon,
                likelihood=QuantileRegression(quantiles=quantiles),
            )
        case "xgboost":
            label = "XGBoost"
            if training_window_type == TrainingWindowType.ROLLING:
                model_class = XGBoostModelRollingWindow
            elif training_window_type == TrainingWindowType.EXPANDING:
                model_class = XGBoostModelExpandingWindow
            else:
                raise ValueError(
                    f"Training window type {training_window_type} not recognized"
                )
            model = model_class(
                lags=24,
                output_chunk_length=max_horizon,
            )
        case "chronos" | "chronos2":
            label = "Chronos2"
            if training_window_type == TrainingWindowType.ROLLING:
                model_class = Chronos2ModelRollingWindow
            elif training_window_type == TrainingWindowType.EXPANDING:
                model_class = Chronos2ModelExpandingWindow
            else:
                raise ValueError(
                    f"Training window type {training_window_type} not recognized"
                )
            model = model_class(
                input_chunk_length=24,
                output_chunk_length=max_horizon,
                likelihood=QuantileRegression(quantiles=quantiles),
            )
        case _:
            raise ValueError(f"Model {model_name} not recognized")
    return label, model
