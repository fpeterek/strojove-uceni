package org.fpeterek.su.clustering

import java.io.File
import scala.io.Source

import org.fpeterek.su.clustering.util.CloseableSource.CloseSource

object DatasetLoader:
  def apply(filename: String): List[Point] =
    Source.fromFile(filename).use { src =>
      src
        .getLines
        .filter(line => line.nonEmpty && !line.forall(_.isSpaceChar))
        .map { line =>
          val split = line.split(';')
          Point(split(0).toDouble, split(1).toDouble)
        }
        .toList
    }
