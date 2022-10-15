package org.fpeterek.su.clustering.util

import scala.io.Source

object CloseableSource:
  implicit class CloseSource(src: Source):
    def use[T](fn: Source => T): T =
      val retval = fn(src)
      src.close()
      retval
