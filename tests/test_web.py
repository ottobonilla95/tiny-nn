from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tiny_nn.network import Network
from tiny_nn.web import make_handler


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Network.build()))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, body=None):
        request = Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json"})
        return urlopen(request, timeout=5)

    def test_page_and_model_info(self):
        with self.request("/") as response:
            self.assertIn(b'canvas id="drawing"', response.read())
        with self.request("/api/info") as response:
            info = json.load(response)
        self.assertGreater(info["hidden_neurons"], 1000)
        self.assertEqual(info["trainable_parameters"], 0)

    def test_generated_example_passes_through_real_model(self):
        with self.request("/api/example?shape=cross&seed=42") as response:
            example = json.load(response)
        with self.request("/api/predict", {"pixels": example["pixels"]}) as response:
            result = json.load(response)
        self.assertEqual(result["label"], "cross")
        self.assertEqual(len(result["template"]), 16)

    def test_invalid_input_returns_400(self):
        for body in ({"pixels": []}, {"pixels": [[2] * 16] * 16}, {"pixels": [[float('nan')] * 16] * 16}, [], {}):
            with self.subTest(body_type=type(body).__name__):
                with self.assertRaises(HTTPError) as caught:
                    self.request("/api/predict", body)
                self.assertEqual(caught.exception.code, 400)
                caught.exception.close()


if __name__ == "__main__":
    unittest.main()
