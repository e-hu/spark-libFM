
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils


/**
 * Created by zrf on 4/18/15.
 */


object TestFM extends App {

  override def main(args: Array[String]): Unit = {
    // 设置Spark运行时的App名称
    val sc = new SparkContext(new SparkConf().setAppName("TESTFM"))
    // 读取训练数据
    val trainData = MLUtils.loadLibSVMFile(sc, "/Users/RecommendSytem/dataset/train.csv").cache()
    // 模型参数
    val task              = args(1).toInt     // 任务数量
    val numIterations     = args(2).toInt     // 迭代次数
    val stepSize          = args(3).toDouble  // 学习步子长
    val miniBatchFraction = args(4).toDouble  // batch最小因子
    // 训练执行
    val fmModel = FMWithSGD.train(
      trainData, 
      task, 
      numIterations，
      stepSize, 
      miniBatchFraction, 
      dim = (true, true, 4), 
      regParam = (0, 0, 0), 
      initStd = 0.1
    )
 }
    
    val fm1 = FMWithSGD.train(training, task = 1, numIterations = 100, stepSize = 0.15, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)
    val fm2 = FMWithLBFGS.train(training, task = 1, numIterations = 20, numCorrections = 5, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)
    
  }
}
