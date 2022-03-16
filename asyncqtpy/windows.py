# mypy: ignore-errors
# © 2018 Gerard Marull-Paretas <gerard@teslabs.com>
# © 2014 Mark Harviston <mark.harviston@gmail.com>
# © 2014 Arve Knudsen <arve.knudsen@gmail.com>
# BSD License

"""Windows specific Quamash functionality."""

import asyncio
import math
import sys
from typing import Optional

try:
    from asyncio import windows_events

    import _overlapped
    import _winapi
except ImportError:  # noqa
    pass  # w/o guarding this import py.test can't gather doctests on platforms w/o _winapi

from . import QtCore, make_signaller
from .common import with_logger

UINT32_MAX = 0xFFFFFFFF


class ProactorEventLoop(asyncio.ProactorEventLoop):

    """Proactor based event loop."""

    def __init__(self):
        super().__init__(IocpProactor())

        self.__event_signaller = make_signaller(list)
        signal = self.__event_signal = self.__event_signaller.signal
        signal.connect(self._process_events)
        self.__event_poller = EventPoller(signal)

    def _process_events(self, events):
        """Process events from proactor."""
        log = self._logger
        for f, callback, transferred, key, ov in events:
            try:
                log.debug(f"Invoking event callback {callback}")
                value = callback(transferred, key, ov)
            except OSError as e:
                log.warning("Event callback failed", exc_info=sys.exc_info())
                if not f.done():
                    f.set_exception(e)
            else:
                if not f.cancelled():
                    f.set_result(value)

    def _before_run_forever(self):
        self.__event_poller.start(self._proactor)

    def _after_run_forever(self):
        self.__event_poller.stop()


@with_logger
class IocpProactor(windows_events.IocpProactor):
    def __init__(self):
        self.__events = []
        super().__init__()
        self._lock = QtCore.QMutex()

    def select(self, timeout: Optional[float] = None):
        """Override in order to handle events in a threadsafe manner."""
        if not self.__events:
            self._poll(timeout)
        tmp = self.__events
        self.__events = []
        return tmp

    def close(self):
        self._logger.debug("Closing")
        super().close()

    def recv(self, conn, nbytes, flags=0):
        with QtCore.QMutexLocker(self._lock):
            return super().recv(conn, nbytes, flags)

    def send(self, conn, buf, flags=0):
        with QtCore.QMutexLocker(self._lock):
            return super().send(conn, buf, flags)

    def _poll(self, timeout=None):
        """Override in order to handle events in a threadsafe manner."""
        if timeout is None:
            ms = UINT32_MAX  # wait for eternity
        elif timeout < 0:
            raise ValueError("negative timeout")
        else:
            # GetQueuedCompletionStatus() has a resolution of 1 millisecond,
            # round away from zero to wait *at least* timeout seconds.
            ms = math.ceil(timeout * 1e3)
            if ms >= UINT32_MAX:
                raise ValueError("timeout too big")

        with QtCore.QMutexLocker(self._lock):
            while True:
                # self._logger.debug('Polling IOCP with timeout {} ms in thread {}...'.format(
                #     ms, threading.get_ident()))
                status = _overlapped.GetQueuedCompletionStatus(self._iocp, ms)
                if status is None:
                    break

                err, transferred, key, address = status
                try:
                    f, ov, obj, callback = self._cache.pop(address)
                except KeyError:
                    # key is either zero, or it is used to return a pipe
                    # handle which should be closed to avoid a leak.
                    if key not in (0, _overlapped.INVALID_HANDLE_VALUE):
                        _winapi.CloseHandle(key)
                    ms = 0
                    continue

                if obj in self._stopped_serving:
                    f.cancel()
                # Futures might already be resolved or cancelled
                elif not f.done():
                    self.__events.append((f, callback, transferred, key, ov))

                ms = 0

    def _wait_for_handle(self, handle, timeout, _is_cancel):
        with QtCore.QMutexLocker(self._lock):
            return super()._wait_for_handle(handle, timeout, _is_cancel)

    def accept(self, listener):
        with QtCore.QMutexLocker(self._lock):
            return super().accept(listener)

    def connect(self, conn, address):
        with QtCore.QMutexLocker(self._lock):
            return super().connect(conn, address)


@with_logger
class EventWorker(QtCore.QThread):
    def __init__(self, proactor, parent):
        super().__init__()

        self.__stop = False
        self.__proactor = proactor
        self.__sig_events = parent.sig_events
        self.__semaphore = QtCore.QSemaphore()

    def start(self):
        super().start()
        self.__semaphore.acquire()

    def stop(self):
        self.__stop = True
        # Wait for thread to end
        self.wait()

    def run(self):
        log = self._logger
        log.debug("Thread started")
        self.__semaphore.release()

        while not self.__stop:
            events = self.__proactor.select(0.01)
            if events:
                log.debug(f"Got events from poll: {events}")
                self.__sig_events.emit(events)

        log.debug("Exiting thread")


@with_logger
class EventPoller:

    """Polling of events in separate thread."""

    def __init__(self, sig_events):
        self.sig_events = sig_events

    def start(self, proactor):
        self._logger.debug(f"Starting (proactor: {proactor})...")
        self.__worker = EventWorker(proactor, self)
        self.__worker.start()

    def stop(self):
        self._logger.debug("Stopping worker thread...")
        self.__worker.stop()
