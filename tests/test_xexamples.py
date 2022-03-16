import asyncio
import json

import pytest

try:
    import aiohttp
except ImportError:
    aiohttp = None

from qtpy.QtCore import Qt

from asyncqtpy import QEventLoop


@pytest.mark.skipif(aiohttp is None, reason="aiohttp is not installed")
def test_aiohttp_example(qtbot):
    from examples.aiohttp_fetch import MainWindow

    loop = QEventLoop()  # App is created by pytest-qt
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    response = ""

    def on_response(*args):
        nonlocal response
        print("Got response!")
        response = window.editResponse.toPlainText()
        print(response)
        window.hide()
        loop.stop()

    window.editResponse.textChanged.connect(on_response)

    loop.call_later(0, qtbot.mouseClick, window.btnFetch, Qt.MouseButton.LeftButton)
    loop.call_later(5, loop.stop)  # Timeout

    with loop:
        loop.run_forever()

    result = json.loads(response)
    assert isinstance(result, dict)


def test_executor_example(qtbot):
    from examples.executor_example import job

    loop = QEventLoop()  # App is created by pytest-qt
    asyncio.set_event_loop(loop)
    with loop:
        loop.run_until_complete(job())
