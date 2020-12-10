import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.{ SparkConf, SparkContext }
import org. apache.spark.rdd.RDD
object Runwordcount {
def main(args: Array[String]):Unit={
Logger.getLogger ("org").setLevel(Level.OFF)
System.setProperty("spark.ui.showConsoleProgress", "false")
println("开始运行RunwordCount")
val sc =new SparkContext(new SparkConf().setAppName ("wordCount").setMaster
("local[4]"))
println("开始读取文本文件...")
val textFile= sc.textFile ("data/wordcount.txt")
println("开始创建RDD...")
val countsRDD = textFile.flatMap(line => line.split(" "))
.map(word-> (word, 1))
.reduceByKey(_ + _)
println("开始保存到文本文件...")
try{
countSRDD.saveAsTextFile("data/output")
println("已经存盘成功")
} catch {
case e: Exception => println("输出目录已经存在,请先删除原有目录");
		}
	}
}
