import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.joda.time._
import org.jfree.data.category.DefaultCategoryDataset

object RunDecisionTreeBinary {
  def main(args: Array[String]): Unit = {
		SetLogger()
		val sc = new SparkContext (new SparkConf().setAppName ( "DecisionTreeBinary").setMaster("local[4]"))
		println ("RunDecisionTreeBinary")
		println("===============数据准备阶段==============")
		val (trainData,validationData, testData,categoriesMap)= PrepareData(sc)
		trainData.persist();validationData.persist(); testData.persist()
		println("===============训练评估阶段==============")
		val model = trainEvaluate(trainData,validationData)
		println("===============测试阶段==============")
		val auc = evaluateModel(model, testData)
		println("使用testdata测试最佳模型，结果 AUC: " + auc)
		println("===============预测数据==============")
		PredictData(sc,model, categoriesMap)
		trainData.unpersist(); validationData.unpersist(); testData.unpersist()
}
  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], Map[String, Int]) = {
  // 导入/转换数据
  print("开始导入数据...")
  val rawDataWithHeader = sc.textFile("data/train.tsv")   // 导入数据
  val rawData = rawDataWithHeader.mapPartitionsWithIndex { // 删除表头
   (idx, iter) => if (idx == 0) iter.drop(1) else iter
  }
  val lines = rawData.map(_.split("\t"))
  println("共计:" + lines.count.toString() + "条")

  // 创建训练评估所需要的数据 RDD[LabeledPoint]
  val categoriesMap = lines.map (fields => fields(3)).distinct.collect.zipWithIndex.toMap
  val labeledPointRDD = lines.map {fields => 
   val trFields = fields.map(_.replaceAll("\"",""))
   val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
   val categoryIdx = categoriesMap(fields(3))
   categoryFeaturesArray(categoryIdx) = 1
   val numericalFeatures = trFields.slice(4, fields.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
   val label = trFields(fields.size - 1).toInt
   LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
  }

  // 以随机方式将数据分为三个部分并且返回
  val Array(trainData, validationData, testData) = labelpointRDD.randomSplit(Array(0.8, 0.1, 0.1))
  println("将数据分成 trainData:" + trainData.count() + ", validationData:" + validationData.count() + ", testData:" + testData.count())
  return (trainData, validationData, testData, categoriesMap)   // 返回数据
 }

 def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
  print("开始训练...")
  val (model, time) = trainModel(trainData, "entropy", 10, 10)
  println("训练完成，所需时间:" + time + "ms")
  val AUC = evaluateModel(model, validationData)
  println("评估结果 AUC=" + AUC)
  return (model)
 }

 def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
  val startTime = new DateTime()
  val model = DecisionTree.trainClassifier(trainData, 2, Map[Int, Int](), impurity, maxDepth, maxBins)
  val endTime = new DateTime()
  val duration = new Duration(startTime, endTime)
  (model, duration.getMillis())   // 返回数据
 }

 def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double) = {
  val scoreAndLabels = validationData.map { data =>
   var predict = model.predict(data.features)
   (predict, data.label)
  }
  val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
  val AUC = Metrics.areaUnderROC
  (AUC)
 }

 def PredictData(sc: SparkContext, model: DecisionTreeModel, categoriesMap:Map[String,Int]): Unit = {
  // 导入并转换数据
  val rawDataWithHeader = sc.textFile("data/test.tsv")
  val rawData = rawDataWithHeader.mapPartitionsWithIndex {
   (idx, iter) => if (idx == 0) iter.drop(1) else iter
  }
  val lines = rawData.map(_.split("\t"))
  println("共计:" + lines.count.toString() + "条")

  // 创建训练评估所需的数据 RDD[LabeledPoint]
  val dataRDD = lines.take(20).map { fields =>
   val trFields = fields.map(_.replaceAll("\"", ""))
   val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
   val categoryIdx = categoriesMap(fields(3))
   categoryFeaturesArray(categoryIdx) = 1
   val numericalFeatures = trFields.slice(4, fields.size).map(d => if (d == "?") 0.0 else d.toDouble)
   val label = 0
   // 进行预测
   val url = trFields(0)
   val Features = Vectors.dense(categoryFeaturesArray ++ numericalFeatures)
   val predict = model.predict(Features).toInt
   var predictDesc = {
    predict match {
     case 0 => "暂时性网页(ephemeral)";
     case 1 => "长青网页(evergreen)";
    }
   }
   println(" 网址:   " + url + "==>预测: " + predictDesc)
  }
 }
}
