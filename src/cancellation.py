"""Cooperative cancellation shared by training, caches and UI workers."""


class Cancelled(RuntimeError):
    pass


def checkpoint(cancel=None):
    if cancel is not None and cancel.is_set():
        raise Cancelled("Cancelled")
