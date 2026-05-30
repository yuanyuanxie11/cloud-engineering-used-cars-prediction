import importlib
import sys
import types
import unittest


def load_handler_with_stubs():
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *_args, **_kwargs: object()

    numpy = types.ModuleType("numpy")
    numpy.asarray = lambda value: value
    numpy.expm1 = lambda value: value

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = lambda records: records

    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda _fh: {}

    features = types.ModuleType("modeling.features")
    features.prepare_xy = lambda df, cfg: (df, [])

    train = types.ModuleType("modeling.train")
    train.prepare_xgb_native_frame = lambda x, cfg: (x, [], [])

    for name, module in {
        "boto3": boto3,
        "numpy": numpy,
        "pandas": pandas,
        "yaml": yaml,
        "modeling.features": features,
        "modeling.train": train,
    }.items():
        sys.modules[name] = module

    sys.modules.pop("serving.handler", None)
    return importlib.import_module("serving.handler")


class ServingHandlerTest(unittest.TestCase):
    def test_options_request_returns_cors_without_loading_model(self):
        handler = load_handler_with_stubs()

        response = handler.handler(
            {"requestContext": {"http": {"method": "OPTIONS", "path": "/predict"}}},
            None,
        )

        self.assertEqual(response["statusCode"], 204)
        self.assertEqual(response["body"], "")
        self.assertEqual(response["headers"]["Access-Control-Allow-Origin"], "*")
        self.assertIn("POST", response["headers"]["Access-Control-Allow-Methods"])

    def test_json_response_includes_cors_headers(self):
        handler = load_handler_with_stubs()

        response = handler._response(400, {"error": "Invalid input"})

        self.assertEqual(response["headers"]["Content-Type"], "application/json")
        self.assertEqual(response["headers"]["Access-Control-Allow-Origin"], "*")


if __name__ == "__main__":
    unittest.main()
