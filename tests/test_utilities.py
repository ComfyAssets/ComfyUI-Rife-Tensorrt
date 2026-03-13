import os
import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError

from utilities import download_file, ColoredLogger


class TestDownloadFile:
    def test_successful_download(self, tmp_path):
        save_path = str(tmp_path / "model.onnx")
        content = b"fake model data" * 100

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": str(len(content))}
        mock_response.iter_content.return_value = [content]

        with patch("utilities.requests.get", return_value=mock_response):
            download_file("https://example.com/model.onnx", save_path)

        assert os.path.exists(save_path)
        with open(save_path, "rb") as f:
            assert f.read() == content

    def test_no_tmp_file_after_success(self, tmp_path):
        save_path = str(tmp_path / "model.onnx")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content.return_value = [b"hello"]

        with patch("utilities.requests.get", return_value=mock_response):
            download_file("https://example.com/model.onnx", save_path)

        assert not os.path.exists(save_path + ".tmp")

    def test_http_error_raises(self, tmp_path):
        save_path = str(tmp_path / "model.onnx")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")

        with patch("utilities.requests.get", return_value=mock_response):
            with pytest.raises(HTTPError):
                download_file("https://example.com/bad.onnx", save_path)

        assert not os.path.exists(save_path)
        assert not os.path.exists(save_path + ".tmp")

    def test_write_error_cleans_tmp(self, tmp_path):
        save_path = str(tmp_path / "model.onnx")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content.side_effect = IOError("disk full")

        with patch("utilities.requests.get", return_value=mock_response):
            with pytest.raises(IOError):
                download_file("https://example.com/model.onnx", save_path)

        assert not os.path.exists(save_path)
        assert not os.path.exists(save_path + ".tmp")

    def test_chunked_download(self, tmp_path):
        save_path = str(tmp_path / "model.onnx")
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        total = sum(len(c) for c in chunks)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-length": str(total)}
        mock_response.iter_content.return_value = chunks

        with patch("utilities.requests.get", return_value=mock_response):
            download_file("https://example.com/model.onnx", save_path)

        with open(save_path, "rb") as f:
            assert f.read() == b"chunk1chunk2chunk3"


class TestColoredLogger:
    def test_creates_logger(self):
        log = ColoredLogger("test-app")
        assert log.logger.name == "test-app"
        assert len(log.logger.handlers) == 1

    def test_all_log_levels(self, capsys):
        log = ColoredLogger("test-levels")
        log.debug("d")
        log.info("i")
        log.warning("w")
        log.error("e")
        log.critical("c")
        output = capsys.readouterr().out
        assert "d" in output
        assert "i" in output
        assert "w" in output
        assert "e" in output
        assert "c" in output

    def test_no_propagation(self):
        log = ColoredLogger("test-no-prop")
        assert log.logger.propagate is False
