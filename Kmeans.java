
package org.apache.hadoop.examples;

import java.io.IOException;
import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import com.google.common.collect.Iterables;
import org.apache.hadoop.conf.Configuration;

/*
K-Means with Eudiean Distance
Use input c2.txt and data.txt
*/
public class Kmeans {
	
	
	private static final int MAX_ITER = 20;
	private static final int K = 10; //The number of centroids 
	private static final int COLS = 58;
	
	
    public static class KmeansMapper
        extends Mapper<Object, Text, Text, Text>{
	
		private Text outputKey = new Text();
		private Text outputValue = new Text();
		
		public void map(Object key, Text value, Context context
               ) throws IOException, InterruptedException {
			/**Get the configuration*/
			Configuration conf = context.getConfiguration();
			/*get data.txt*/
			String data = value.toString();
			String[] vals = data.split(" ");
			/*determine which centroid is closet*/
			double min = -1;
			int Id = -1; 
			for(int i=0; i<K; i++){
				String centroid = conf.get(Integer.toString(i));
				String[] cen = centroid.split(" ");
				/*calculate Mahattan Distance*/
				double cost = 0;
				for(int j=0; j<COLS;j++){
					cost = cost + Math.abs(Double.parseDouble(vals[j]) - Double.parseDouble(cen[j]));
				}
				if(cost < min || min == -1){
					min = cost;
					Id = i;
				}
			}
			/*Emit the key and value*/
			outputKey.set(Integer.toString(Id));
			outputValue.set(data + "," + Double.toString(min));
			context.write(outputKey, outputValue);
		}
	}

	public static class KmeansReducer
        extends Reducer<Text,Text,Text,Text> {
		private MultipleOutputs<Text, Text> mos;
	
    public void reduce(Text key, Iterable<Text> values,
                        Context context
                        ) throws IOException, InterruptedException {
		double[] Total = new double[COLS];
		for(int i=0; i<COLS; i++){
			Total[i] = 0;
		}
		int size = 0;
		double totalCost = 0;
		for(Text value: values){
			/*
			data[0] = the point belongs to this centroid
			data[1] = the cost of the point and this centroid
			*/
			String[] data = value.toString().split(",");
			String[] point = data[0].split(" ");
			for(int i=0; i<COLS; i++){
				Total[i] += Double.parseDouble(point[i]);
			}
			totalCost += Double.parseDouble(data[1]);
			size++;
		}
		/*Calculate new centroid*/
		String result = "";
		for(int i=0; i<COLS; i++){
			Total[i] = Total[i] / size;
			result = result + Double.toString(Total[i]) + " ";
		}
		context.write(null, new Text("cost: " + key.toString() + "," + Double.toString(totalCost) + "," + result));
	}
}


public static void main(String[] args) throws Exception {

	Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 2) {
        System.err.println("Usage: matrix multiplication <in> <out>");
        System.exit(2);
    }
	/*Declare FileSystem*/
	FileSystem fs = FileSystem.get(conf);
	
	for(int i=0; i<MAX_ITER; i++){
		/**
		Input:
		In the first iteration, the inputfile:
			centroid: /user/data/c1.txt
			data: /user/data/data.txt
		In the upcoming iteration, the inputfile:
			data: /user/data/data.txt
		Output:
		Set the outputfile:
			//each node belongs to which centroid (number from 0-9)
			cost: /user/output/{iteration+1}/cost.txt
		**/
		/*
		PS. We don't have to save the centroid files, since we save it in the configuration
		*/
		/*
		put these ten centroid value in the configuration
		so that we can use it in mapper
		*/
		
		/**
		Design of mapper and reducer
		mapper:
		calculate the cost value of every value and centroid points
		use the centroid id as key, and the vector of this node as value and the cost value in the final position
		{K , V} = { 'ID', '0.2 0.16 .... 0,'cost''}
		
		reducer:
		since the key is Centroid ID, we will automatically get the clustered points
		Then we can calculate new ten centroid points.
		Put it in the configuration.
		The only thing we have to output is the cost value
		**/
		String filename = "";
		if(i==0){
			/*In the first iteration, read the c2.txt and set it in the configuration*/
			/*Read file*/
			filename = "/user/root/data/c2.txt";
			Path CPath = new Path(filename);
			InputStream is = fs.open(CPath);
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			String line = br.readLine();
			int k = 0; // id of centroid
			while (line != null){
			/*set the configuration*/
				conf.set(Integer.toString(k), line);
				line = br.readLine();
				k++;
			}
		}
		
		
		Job job = new Job(conf, "K-Means");
		job.setJarByClass(Kmeans.class);
		job.setMapperClass(KmeansMapper.class);
		job.setReducerClass(KmeansReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.getConfiguration().set("mapreduce.output.basename", "tmp");
		
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1] + "/" + Integer.toString(i+1)));
		job.waitForCompletion(true) ;
		
		/**
		Calculate the total cost and the new centroid
		First, read the tmp file
		**/
		/*In the upcoming iteration, we set conf here and write the outputfile simultaneously*/
		Path TmpPath = new Path(otherArgs[1] + "/" + Integer.toString(i+1) + "/tmp-r-00000");
		InputStream is = fs.open(TmpPath);
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String line = br.readLine();
		int k = 0; // id of centroid
		double totalCost = 0;
		while (line != null){
			/*set the configuration*/
			String[] ents = line.split(","); 
			line = ents[2];
			totalCost += Double.parseDouble(ents[1]);
			conf.set(Integer.toString(k), line);
			line = br.readLine();
			k++;
		}
		Path CostPath= new Path( otherArgs[1] + "/" + Integer.toString(i+1) + "/cost");
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs.create(CostPath, true), "UTF-8"));
		bw.write(Double.toString(totalCost) + "\n");
		bw.flush();
		bw.close();
		
		Path OutPath= new Path( otherArgs[1] + "/" + Integer.toString(i+1) + "/centroid");
		BufferedWriter bw2 = new BufferedWriter(new OutputStreamWriter(fs.create(OutPath, true), "UTF-8"));
		for(int j=0; j<K; j++){
			bw2.write(conf.get(Integer.toString(j)) + "\n");
		}
		bw2.flush();
		bw2.close();
	}
	/*Read the final centroid points and calculate their Euclidean distance and Mahattan distance*/
	Path FinalPath = new Path(otherArgs[1] + "/" + Integer.toString(MAX_ITER) + "/centroid");
	InputStream is = fs.open(FinalPath);
	BufferedReader br = new BufferedReader(new InputStreamReader(is));
	double[][] centroidPoints = new double[K][COLS];
	String line = br.readLine();
	int k=0;
	while (line != null){
		String[] val = line.split(" "); 
		if(k < K){
			for(int i=0; i<COLS; i++){
				centroidPoints[k][i] = Double.parseDouble(val[i]);
			}
			k++;
		}
		line = br.readLine();
	}
	/*Write*/
	Path EPath= new Path( otherArgs[1] + "/EuclideanDistance");
	BufferedWriter ebw = new BufferedWriter(new OutputStreamWriter(fs.create(EPath, true), "UTF-8"));
	Path MPath= new Path( otherArgs[1] + "/ManhattanDistance");
	BufferedWriter mbw = new BufferedWriter(new OutputStreamWriter(fs.create(MPath, true), "UTF-8"));
	for(int i=0; i<K; i++){
		for(int j=i+1; j<K; j++){
			ebw.write(Integer.toString(i) + "&" + Integer.toString(j) + ": ");
			mbw.write(Integer.toString(i) + "&" + Integer.toString(j) + ": ");
			double edis = 0;
			double mdis = 0;
			for(int v=0; v<COLS; v++){
				edis += Math.pow(centroidPoints[i][v] - centroidPoints[j][v], 2);
				mdis += Math.abs(centroidPoints[i][v] - centroidPoints[j][v]);
			}
			edis = Math.sqrt(edis);
			ebw.write(Double.toString(edis) + "\n");
			mbw.write(Double.toString(mdis) + "\n");
		}
	}
	ebw.flush();
	ebw.close();
	mbw.flush();
	mbw.close();
}
}




