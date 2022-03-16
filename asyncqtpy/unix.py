# © 2018 Gerard Marull-Paretas <gerard@teslabs.com>
# © 2014 Mark Harviston <mark.harviston@gmail.com>
# © 2014 Arve Knudsen <arve.knudsen@gmail.com>
# BSD License

"""UNIX specific Quamash functionality."""
import asyncio
import collections
import logging
import selectors
from typing import Iterator, Optional, Protocol, Union

from . import QtCore, with_logger

QSocketNotifier = QtCore.QSocketNotifier

EVENT_READ = 1 << 0
EVENT_WRITE = 1 << 1


class HasFileno(Protocol):
    def fileno(self) -> int:
        ...


FileObj = Union[int, HasFileno]


def _fileobj_to_fd(fileobj: Union[int, HasFileno, selectors.SelectorKey]) -> int:
    """
    Return a file descriptor from a file object.

    Parameters:
    fileobj -- file object or file descriptor

    Returns:
    corresponding file descriptor

    Raises:
    ValueError if the object is invalid

    """
    if isinstance(fileobj, int):
        fd = fileobj
    elif isinstance(fileobj, selectors.SelectorKey):
        fd = fileobj.fd
    else:
        try:
            fd = int(fileobj.fileno())
        except (AttributeError, TypeError, ValueError) as ex:
            raise ValueError(f"Invalid file object: {fileobj!r}") from ex
    if fd < 0:
        raise ValueError(f"Invalid file descriptor: {fd}")
    return fd


class SelectorMapping(collections.abc.Mapping):

    """Mapping of file objects to selector keys."""

    def __init__(self, selector: "Selector"):
        self._selector = selector

    def __len__(self) -> int:
        return len(self._selector._fd_to_key)

    def __getitem__(self, fileobj: FileObj) -> selectors.SelectorKey:
        try:
            fd = self._selector._fileobj_lookup(fileobj)
            return self._selector._fd_to_key[fd]
        except KeyError:
            raise KeyError(f"{fileobj!r} is not registered") from None

    def __iter__(self) -> Iterator[FileObj]:
        return iter(self._selector._fd_to_key)


@with_logger
class Selector(selectors.BaseSelector):
    _logger: logging.Logger

    def __init__(self, parent):
        # this maps file descriptors to keys
        self._fd_to_key: dict[FileObj, selectors.SelectorKey] = {}
        # read-only mapping returned by get_map()
        self.__map = SelectorMapping(self)
        self.__read_notifiers: dict[int, QtCore.QSocketNotifier] = {}
        self.__write_notifiers: dict[int, QtCore.QSocketNotifier] = {}
        self.__parent = parent

    def select(self, *args, **kwargs):
        """Implement abstract method even though we don't need it."""
        raise NotImplementedError

    def _fileobj_lookup(
        self, fileobj: Union[selectors.SelectorKey, int, HasFileno]
    ) -> int:
        """Return a file descriptor from a file object.

        This wraps _fileobj_to_fd() to do an exhaustive search in case
        the object is invalid but we still have it in our map.  This
        is used by unregister() so we can unregister an object that
        was previously registered even if it is closed.  It is also
        used by _SelectorMapping.
        """
        try:
            return _fileobj_to_fd(fileobj)
        except ValueError:
            # Do an exhaustive search.
            for key in self._fd_to_key.values():
                if key.fileobj is fileobj:
                    return key.fd
            # Raise ValueError after all.
            raise

    def register(
        self, fileobj: FileObj, events: int, data: Optional[bytes] = None
    ) -> selectors.SelectorKey:
        if (not events) or (events & ~(EVENT_READ | EVENT_WRITE)):
            raise ValueError(f"Invalid events: {events!r}")

        key = selectors.SelectorKey(
            fileobj, self._fileobj_lookup(fileobj), events, data
        )
        fd = key.fd
        if fd in self._fd_to_key:
            raise KeyError(f"{fileobj!r} (FD {fd}) is already registered")

        self._fd_to_key[fd] = key

        if events & EVENT_READ:
            notifier = QSocketNotifier(fd, QSocketNotifier.Read)
            notifier.activated.connect(self.__on_read_activated)
            self.__read_notifiers[fd] = notifier
        if events & EVENT_WRITE:
            # TODO: This should pause
            notifier = QSocketNotifier(fd, QSocketNotifier.Write)
            notifier.activated.connect(self.__on_write_activated)
            self.__write_notifiers[fd] = notifier

        return key

    def __on_read_activated(self, fd: int):
        # self._logger.debug(f"File {fd} ready to read")
        key = self._key_from_fd(fd)
        if key:
            self.__parent._process_event(key, EVENT_READ & key.events)

    def __on_write_activated(self, fd: int):
        # On python 3.10 this fires continuously...
        # self._logger.debug(f"File {fd} ready to write")
        key = self._key_from_fd(fd)
        if key:
            self.__parent._process_event(key, EVENT_WRITE & key.events)
            self._pause_writer(fd)

    def _pause_writer(self, fd: int, timeout: float = 0.1):
        # Pause write callbacks for a few ms to avoid high cpu usage
        notifier = self.__write_notifiers[fd]
        notifier.setEnabled(False)
        loop = asyncio.get_event_loop()
        loop.call_later(timeout, self._resume_writer, fd)

    def _resume_writer(self, fd: int):
        try:
            notifier = self.__write_notifiers[fd]
            notifier.setEnabled(True)
        except KeyError:
            pass

    def unregister(self, fileobj: FileObj):
        def drop_notifier(notifiers: dict[int, QtCore.QSocketNotifier]):
            try:
                notifier = notifiers.pop(key.fd)
            except KeyError:
                pass
            else:
                notifier.activated.disconnect()  # type: ignore
                del notifier

        try:
            key = self._fd_to_key.pop(self._fileobj_lookup(fileobj))
        except KeyError:
            raise KeyError(f"{fileobj!r} is not registered") from None

        drop_notifier(self.__read_notifiers)
        drop_notifier(self.__write_notifiers)

        return key

    def modify(
        self, fileobj: FileObj, events: int, data: Optional[bytes] = None
    ) -> selectors.SelectorKey:
        try:
            key = self._fd_to_key[self._fileobj_lookup(fileobj)]
        except KeyError:
            raise KeyError(f"{fileobj!r} is not registered") from None
        if events != key.events:
            self.unregister(fileobj)
            key = self.register(fileobj, events, data)
        elif data != key.data:
            # Use a shortcut to update the data.
            key = key._replace(data=data)
            self._fd_to_key[key.fd] = key
        return key

    def close(self):
        self._logger.debug("Closing")
        self._fd_to_key.clear()
        self.__read_notifiers.clear()
        self.__write_notifiers.clear()

    def get_map(self):
        return self.__map

    def _key_from_fd(self, fd: int) -> Optional[selectors.SelectorKey]:
        """
        Return the key associated to a given file descriptor.

        Parameters:
        fd -- file descriptor

        Returns:
        corresponding key, or None if not found

        """
        try:
            return self._fd_to_key[fd]
        except KeyError:
            return None


class SelectorEventLoop(asyncio.SelectorEventLoop):
    _logger: logging.Logger

    def __init__(self):
        self._signal_safe_callbacks = []
        self._closed = False

        selector = Selector(self)
        asyncio.SelectorEventLoop.__init__(self, selector)

    def _before_run_forever(self):
        pass

    def _after_run_forever(self):
        pass

    def _process_event(self, key: selectors.SelectorKey, mask: int):
        """Selector has delivered us an event."""
        # log = self._logger
        # log.debug(f"Processing event with key {key} and mask {mask}")
        fileobj, (reader, writer) = key.fileobj, key.data
        if mask & selectors.EVENT_READ and reader is not None:
            if reader._cancelled:
                self.remove_reader(fileobj)
            else:
                # log.debug(f"Invoking reader callback: {reader}")
                reader._run()
        if mask & selectors.EVENT_WRITE and writer is not None:
            if writer._cancelled:
                self.remove_writer(fileobj)
            else:
                # log.debug(f"Invoking writer callback: {writer}")
                writer._run()
