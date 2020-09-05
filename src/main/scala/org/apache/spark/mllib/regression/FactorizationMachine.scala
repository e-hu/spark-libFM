package org.apache.spark.mllib.regression

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.util.Random

import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.optimization.{Updater, Gradient}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.Loader._
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Created by zrf on 4/13/15.
  */

/**
  * Factorization Machine model.
  */
class FMModel(val task: Int,
              val factorMatrix: Matrix,
              val weightVector: Option[Vector],
              val intercept: Double,
              val min: Double,
              val max: Double) extends Serializable with Saveable {
  // 特征数量与样本数量
  val numFeatures = factorMatrix.numCols
  val numFactors = factorMatrix.numRows
  require(numFeatures > 0 && numFactors > 0)
  require(task == 0 || task == 1)
  // 使用测试数据进行模型预测
  def predict(testData: Vector): Double = {
    require(testData.size == numFeatures)
    var pred = intercept
    if (weightVector.isDefined) {
      testData.foreachActive {
        case (i, v) =>
          pred += weightVector.get(i) * v
      }
    }
    for (f <- 0 until numFactors) {
      var sum = 0.0
      var sumSqr = 0.0
      testData.foreachActive {
        case (i, v) =>
          val d = factorMatrix(f, i) * v
          sum += d
          sumSqr += d * d
      }
      pred += (sum * sum - sumSqr) / 2
    }
    task match {
      case 0 =>
        Math.min(Math.max(pred, min), max)
      case 1 =>
        1.0 / (1.0 + Math.exp(-pred))
    }
  }
  // 模型保存
  override def save(sc: SparkContext, path: String): Unit = {
    val data = FMModel.SaveLoadV1_0.Data(factorMatrix, weightVector, intercept, min, max, task)
    FMModel.SaveLoadV1_0.save(sc, path, data)
  }
}

object FMModel extends Loader[FMModel] {
  // 模型版本
  private object SaveLoadV1_0 {
    def thisFormatVersion = "1.0"
    def thisClassName = "org.apache.spark.mllib.regression.FMModel"
    // 模型数据导入或导出
    case class Data(factorMatrix: Matrix, weightVector: Option[Vector], intercept: Double,
                    min: Double, max: Double, task: Int)
    // 模型保存函数
    def save(sc: SparkContext, path: String, data: Data): Unit = {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      // 创建模型的JSON元数据
      val metadata = compact(render(
          ("class" -> this.getClass.getName) ~ 
          ("version" -> thisFormatVersion) ~
          ("numFeatures" -> data.factorMatrix.numCols) ~ 
          ("numFactors" -> data.factorMatrix.numRows) ~ 
          ("min" -> data.min) ~ 
          ("max" -> data.max) ~ 
          ("task" -> data.task)
      ))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(metadataPath(path))
      // 创建模型的Parquet数据
      val dataRDD: DataFrame = sc.parallelize(Seq(data), 1).toDF()
      dataRDD.saveAsParquetFile(dataPath(path))
    }
   
    // 加载模型函数
    def load(sc: SparkContext, path: String): FMModel = {
      val sqlContext = new SQLContext(sc)
      // 加载Parquet数据
      val dataRDD = sqlContext.parquetFile(dataPath(path))
      // 检测Schema信息是否正确
      checkSchema[Data](dataRDD.schema)
      val dataArray = dataRDD.select("task", "factorMatrix", "weightVector", "intercept", "min", "max").take(1)
      assert(dataArray.length == 1, s"无法从 ${dataPath(path)} 加载FMModel模型数据")
      val data = dataArray(0)
      val task = data.getInt(0)
      val factorMatrix = data.getAs[Matrix](1)
      val weightVector = data.getAs[Option[Vector]](2)
      val intercept = data.getDouble(3)
      val min = data.getDouble(4)
      val max = data.getDouble(5)
      new FMModel(task, factorMatrix, weightVector, intercept, min, max)
    }
  }
}

// 因子向量机梯度类
class FMGradient(val task: Int, val k0: Boolean, val k1: Boolean, val k2: Int,
                 val numFeatures: Int, val min: Double, val max: Double) extends Gradient {
  // 预测函数
  private def predict(data: Vector, weights: Vector): (Double, Array[Double]) = {
    var pred = if (k0) weights(weights.size - 1) else 0.0
    // 特征k序列
    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          pred += weights(pos + i) * v
      }
    }
    val sum = Array.fill(k2)(0.0)
    for (f <- 0 until k2) {
      var sumSqr = 0.0
      data.foreachActive {
        case (i, v) =>
          val d = weights(i * k2 + f) * v
          sum(f) += d
          sumSqr += d * d
      }
      pred += (sum(f) * sum(f) - sumSqr) / 2
    }
    if (task == 0) {
      pred = Math.min(Math.max(pred, min), max)
    }
    (pred, sum)
  }
 
  // 计算梯度
  private def cumulateGradient(data: Vector, weights: Vector,
                               pred: Double, label: Double,
                               sum: Array[Double], cumGrad: Vector): Unit = {
    val mult = task match {
      case 0 =>
        pred - label
      case 1 =>
        -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))
    }
    cumGrad match {
      case vec: DenseVector =>
        val cumValues = vec.values

        if (k0) {
          cumValues(cumValues.length - 1) += mult
        }

        if (k1) {
          val pos = numFeatures * k2
          data.foreachActive {
            case (i, v) =>
              cumValues(pos + i) += v * mult
          }
        }

        data.foreachActive {
          case (i, v) =>
            val pos = i * k2
            for (f <- 0 until k2) {
              cumValues(pos + f) += (sum(f) * v - weights(pos + f) * v * v) * mult
            }
        }
    }
  }

  // 计算样本损失值
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val cumGradient = Vectors.dense(Array.fill(weights.size)(0.0))
    val loss = compute(data, label, weights, cumGradient)
    (cumGradient, loss)
  }
 
  // 计算梯度值
  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    require(data.size == numFeatures)
    val (pred, sum) = predict(data, weights)
    cumulateGradient(data, weights, pred, label, sum, cumGradient)

    task match {
      case 0 =>
        (pred - label) * (pred - label)
      case 1 =>
        1 - Math.signum(pred * label)
    }
  }
}

// 因子向量机的更新器（迭代）
class FMUpdater(val k0: Boolean, val k1: Boolean, val k2: Int,
                val r0: Double, val r1: Double, val r2: Double,
                val numFeatures: Int) extends Updater {
  override def compute(weightsOld: Vector, gradient: Vector,
                       stepSize: Double, iter: Int, regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / math.sqrt(iter)
    val len = weightsOld.size
    val weightsNew = Array.fill(len)(0.0)
    var regVal = 0.0
   
    if (k0) {
      weightsNew(len - 1) = weightsOld(len - 1) - thisIterStepSize * (gradient(len - 1) + r0 * weightsOld(len - 1))
      regVal += r0 * weightsNew(len - 1) * weightsNew(len - 1)
    }
    
    if (k1) {
      for (i <- numFeatures * k2 until numFeatures * k2 + numFeatures) {
        weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r1 * weightsOld(i))
        regVal += r1 * weightsNew(i) * weightsNew(i)
      }
    }

    for (i <- 0 until numFeatures * k2) {
      weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r2 * weightsOld(i))
      regVal += r2 * weightsNew(i) * weightsNew(i)
    }

    (Vectors.dense(weightsNew), regVal / 2)
  }
}
