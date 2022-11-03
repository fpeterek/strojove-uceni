package org.fpeterek.su.clustering

import java.io.File
import org.fpeterek.su.clustering.distance.DistanceMetric


object Main:

  private def processFile(file: String): String =
    File(file).exists() match
      case false => throw RuntimeException(s"File $file does not exist")
      case true  => file

  private def processLinkage(linkage: String) = linkage.trim.toLowerCase match
    case "single"   => Linkage.Single
    case "complete" => Linkage.Complete

  private def processMetric(metric: String) = metric.trim.toLowerCase match
    case "euclidean" => DistanceMetric.Euclidean
    case "manhattan" => DistanceMetric.Manhattan

  private def processClusterCount(count: String) = count.trim.forall(_.isDigit) match
    case false => throw RuntimeException("Number of clusters must be a number")
    case true  => count.toInt match
      case 0 => throw RuntimeException("Number of clusters must be > 0")
      case x => x

  private def outputFileName(infile: String, linkage: Linkage, metric: DistanceMetric, clusters: Int): String =
    val begin = infile.lastIndexOf("/") match
      case -1 => 0
      case x  => x+1
    val in = infile.slice(begin, infile.lastIndexOf("."))
    s"out/${in}_${linkage}_${metric}_$clusters.json"

  private def process(inputFile: String, linkage: Linkage, metric: DistanceMetric, numClusters: Int): Unit =
    println(s"Processing: $inputFile, $linkage, $metric, $numClusters")
    val points = DatasetLoader(inputFile)
    val clusters = Clustering(linkage, metric, numClusters)(points)
    val outfile = outputFileName(inputFile, linkage, metric, numClusters)
    JsonWriter(clusters, outfile)

  private def processOne(args: Array[String]): Unit =
    val file       = processFile(args(0))
    val linkage    = processLinkage(args(1))
    val metric     = processMetric(args(2))
    val clusters   = processClusterCount(args(3))

    process(file, linkage, metric, clusters)

  private def dataFiles = File("data/").listFiles.map(_.getPath)

  private def processPredefined(): Unit =
    dataFiles.foreach { file =>
      Linkage.values.foreach { linkage =>
        DistanceMetric.values.foreach { metric =>
          Seq(2, 3, 5).foreach { clusterCount =>
            process(file, linkage, metric, clusterCount)
          }
        }
      }
    }

  def main(args: Array[String]): Unit = args.length match
    case 4 => processOne(args)
    case _ => processPredefined()
