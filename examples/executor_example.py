import asyncio
import sys
import time

from qtpy.QtWidgets import QApplication, QProgressBar

from asyncqt import QEventLoop, QThreadExecutor

app = QApplication(sys.argv)
loop = QEventLoop(app)
asyncio.set_event_loop(loop)

progress = QProgressBar()
progress.setRange(0, 99)
progress.show()


async def master():
    await first_50()
    with QThreadExecutor(1) as exec:
        await loop.run_in_executor(exec, last_50)


async def first_50():
    for i in range(50):
        progress.setValue(i)
        await asyncio.sleep(0.1)


def last_50():
    for i in range(50, 100):
        loop.call_soon_threadsafe(progress.setValue, i)
        time.sleep(0.1)


with loop:
    loop.run_until_complete(master())
