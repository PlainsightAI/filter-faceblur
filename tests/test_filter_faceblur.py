#!/usr/bin/env python

import logging
import multiprocessing
import os
import sys
import unittest

from filter_faceblur.filter import FilterFaceblur

logger = logging.getLogger(__name__)

logger.setLevel(int(getattr(logging, (os.getenv('LOG_LEVEL') or 'INFO').upper())))

VERBOSE   = '-v' in sys.argv or '--verbose' in sys.argv
LOG_LEVEL = logger.getEffectiveLevel()


class TestFilterFaceblur(unittest.TestCase):
    def test_filter_faceblur(self):
        if VERBOSE and LOG_LEVEL <= logging.WARNING:
            print()

        # TODO: test here

        pass


if __name__ != '__mp_main__':
    multiprocessing.set_start_method('spawn')  # CUDA doesn't like fork()

if __name__ == '__main__':
    unittest.main()
