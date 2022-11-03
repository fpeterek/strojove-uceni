package org.fpeterek.su.clustering

import org.json.{JSONArray, JSONObject}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

object JsonWriter:

  private def pointToJson(point: Point) = JSONObject()
    .put("x", point.x)
    .put("y", point.y)

  private def clusterToArr(cluster: Cluster) =
    val arr = JSONArray()
    cluster.points
      .map(pointToJson)
      .foreach(arr.put)

    arr

  private def createClusterArray(clusters: Seq[Cluster]) =
    val arr = JSONArray()
    clusters
      .map(clusterToArr)
      .foreach(arr.put)

    arr

  def apply(clusters: Seq[Cluster], outfile: String): Unit =
    val result = JSONObject()
    result
      .put("clusters", createClusterArray(clusters))

    Files.write(Paths.get(outfile), result.toString.getBytes(StandardCharsets.UTF_8))
