import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.hadoop.mapred.jobcontrol.JobControl;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

/**
 * KnapsackBruteForce uses a brute force way to find the optimal solution for the knapsack problem.
 * It splits the work load to multiple mappers by assigning each mapper to search a best solution
 * within a part of the solution space.
 *
 */
public class KnapsackBruteForce extends Configured implements Tool {

	public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, DoubleWritable, Text> {

		private FileSystem fileSystem;
		private List<String> splitFileList;
		private List<Double> thresholdList;
		private List<String> knapsackList;
		private double minimumSumValue;
		private int leafLevel;

		@Override
		public void configure(JobConf jobConf) {
			BufferedReader inputBufferedReader = null;

			try {
				fileSystem = FileSystem.get(jobConf);
				splitFileList = new ArrayList<String>();
				thresholdList = new ArrayList<Double>();
				knapsackList = new ArrayList<String>();
				minimumSumValue = Double.MAX_VALUE;
				leafLevel = 3;

				inputBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
						"knapsack_input.txt"))));

				String inputKeyLine = inputBufferedReader.readLine();

				// initialize splitFileList with the split file locations, note the split0 file is
				// not included, because it is used as the input file of all the mappers
				if (inputKeyLine != null) {
					String[] keyArray = inputKeyLine.split("\t");

					for (int i = 1; i < keyArray.length; i++) {
						splitFileList.add("knapsack_split" + i + ".txt");
					}
				}

				String inputThresholdLine = inputBufferedReader.readLine();

				if (inputThresholdLine != null) {
					String[] thresholdArray = inputThresholdLine.split("\t");

					for (int i = 0; i < thresholdArray.length; i++) {
						thresholdList.add(Double.parseDouble(thresholdArray[i]));
					}
				}

			} catch (IOException e) {
				e.printStackTrace();

			} finally {
				try {
					if (inputBufferedReader != null) {
						inputBufferedReader.close();
					}

				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		public void map(LongWritable key, Text value, OutputCollector<DoubleWritable, Text> outputCollector,
				Reporter reporter) throws IOException {

			// use each line of the split0 file as the first object added to the knapsack and start to search
			String[] splitArray = value.toString().split("\t");
			List<Double> valueList = new ArrayList<Double>();

			for (int i = 1; i < splitArray.length; i++) {
				valueList.add(Double.parseDouble(splitArray[i]));
			}

			// knapsackPathList is the list which maintains the content of the knapsack
			List<String> knapsackPathList = new ArrayList<String>();
			knapsackPathList.add(value.toString());

			// use recursive method to search the solution space
			addNextLevelValue(0, valueList, knapsackPathList);

			if (!knapsackList.isEmpty()) {
				String knapsackListValue = "";

				for (int i = 0; i < knapsackList.size(); i++) {
					if (i != 0) {
						knapsackListValue += "-";
					}
					knapsackListValue += knapsackList.get(i);
				}

				// output the final minimum sum value and the string representation of the knapsack content
				outputCollector.collect(new DoubleWritable(minimumSumValue), new Text(knapsackListValue));
			}
		}

		/**
		 * a recursive method to depth first search the optimal solution for the knapsack problem
		 * 
		 * @param level
		 * 		different level has different key for the search method to find a object with 
		 * @param valueList
		 * 		valueList maintains the values accumulated from the previous levels
		 * @param knapsackPathList
		 * 		knapsackPathList maintains the knapsack content till the current level
		 */
		private void addNextLevelValue(int level, List<Double> valueList, List<String> knapsackPathList) {
			BufferedReader splitFileBufferedReader = null;
			String knapsackLine = null;

			try {
				// read the split file according to the current level
				splitFileBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
						splitFileList.get(level)))));

				// test each candidate object from the split file
				while ((knapsackLine = splitFileBufferedReader.readLine()) != null) {
					String[] splitArray = knapsackLine.split("\t");
					List<Double> newValueList = new ArrayList<Double>();

					// add the values of the current object with the previous values
					for (int i = 1; i < splitArray.length; i++) {
						newValueList.add(Double.parseDouble(splitArray[i]) + valueList.get(i - 1));
					}

					// if the leaf level is reached, test the constraints
					if (level == leafLevel) {
						boolean allConstraintsSatisfied = true;

						for (int i = 0; i < newValueList.size(); i++) {
							if (newValueList.get(i) < thresholdList.get(i)) {
								allConstraintsSatisfied = false;
								break;
							}
						}

						if (allConstraintsSatisfied) {
							double newSumValue = 0.0;

							for (int i = 0; i < newValueList.size(); i++) {
								newSumValue += newValueList.get(i);
							}

							// compare the current result with the global optimal result,
							// if it's better, replace the global optimal result with it
							if (newSumValue < minimumSumValue) {
								minimumSumValue = newSumValue;
								knapsackList.clear();

								for (String knapsackPath : knapsackPathList) {
									knapsackList.add(new String(knapsackPath));
								}
								knapsackList.add(new String(knapsackLine));
							}
						}

					} else {
						// if it is not the leaf level, calculate the current result also
						double currentSumValue = 0.0;

						for (int i = 0; i < newValueList.size(); i++) {
							currentSumValue += newValueList.get(i);
						}

						// prune the branch of the search tree, if the current result is
						// already bigger than the global optimal result
						if (currentSumValue < minimumSumValue) {
							knapsackPathList.add(knapsackLine);

							// start the next level of search
							addNextLevelValue(level + 1, newValueList, knapsackPathList);

							knapsackPathList.remove(knapsackLine);
						}
					}
				}

			} catch (IOException ioe) {
				ioe.printStackTrace();

			} finally {
				try {
					if (splitFileBufferedReader != null) {
						splitFileBufferedReader.close();
					}

				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static class Reduce extends MapReduceBase implements Reducer<DoubleWritable, Text, DoubleWritable, Text> {

		public void reduce(DoubleWritable key, Iterator<Text> values,
				OutputCollector<DoubleWritable, Text> outputCollector, Reporter reporter) throws IOException {

			// just output all the data received from mapper, this reduce class is needed to ensure a sorted output
			while (values.hasNext()) {
				outputCollector.collect(key, values.next());
			}
		}
	}

	public int run(String[] args) throws Exception {
		// this is a job pipeline with two jobs
		JobControl jobControl = new JobControl("KnapsackBruteForce");

		JobConf datasetSpliterJobConf = new JobConf(DatasetSpliter.class);
		datasetSpliterJobConf.setJobName("DatasetSpliter");

		datasetSpliterJobConf.setOutputKeyClass(Text.class);
		datasetSpliterJobConf.setOutputValueClass(Text.class);

		datasetSpliterJobConf.setMapperClass(DatasetSpliter.Map.class);
		datasetSpliterJobConf.setReducerClass(DatasetSpliter.Reduce.class);

		datasetSpliterJobConf.setInputFormat(TextInputFormat.class);
		datasetSpliterJobConf.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(datasetSpliterJobConf, "knapsack_dataset.txt");
		FileOutputFormat.setOutputPath(datasetSpliterJobConf, new Path("knapsack_split"));

		// the first job is to split the data
		Job datasetSpliterJob = new Job(datasetSpliterJobConf);

		jobControl.addJob(datasetSpliterJob);

		JobConf bruteForceJobConf = new JobConf(KnapsackBruteForce.class);
		bruteForceJobConf.setJobName("KnapsackBruteForce");

		bruteForceJobConf.setOutputKeyClass(DoubleWritable.class);
		bruteForceJobConf.setOutputKeyComparatorClass(DoubleWritable.Comparator.class);
		bruteForceJobConf.setOutputValueClass(Text.class);

		bruteForceJobConf.setMapperClass(KnapsackBruteForce.Map.class);
		bruteForceJobConf.setReducerClass(KnapsackBruteForce.Reduce.class);

		bruteForceJobConf.setInputFormat(TextInputFormat.class);
		bruteForceJobConf.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(bruteForceJobConf, "knapsack_split0.txt");
		FileOutputFormat.setOutputPath(bruteForceJobConf, new Path("knapsack_brute_force"));

		// the second job is to do the brute force search
		Job bruteForceJob = new Job(bruteForceJobConf);
		bruteForceJob.addDependingJob(datasetSpliterJob);

		jobControl.addJob(bruteForceJob);

		new Thread(jobControl).start();

		while (!jobControl.allFinished()) {
			Thread.sleep(2000);
		}

		jobControl.stop();

		BufferedReader bruteForceBufferedReader = null;
		PrintWriter bruteForcePrintWriter = null;
		String bruteForceResult = null;

		try {
			FileSystem fileSystem = FileSystem.get(getConf());

			bruteForceBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
					"knapsack_brute_force/part-00000"))));
			bruteForcePrintWriter = new PrintWriter(fileSystem.create(new Path("knapsack_bf_output.txt")));

			// in the end, select the first result of the brute force search and output it as the final solution
			if ((bruteForceResult = bruteForceBufferedReader.readLine()) != null) {
				int resultIndex = bruteForceResult.indexOf("\t");
				bruteForcePrintWriter.println(String.format("%.2f", Double.parseDouble(bruteForceResult
						.substring(0, resultIndex))));

				String[] knapsackArray = bruteForceResult.substring(resultIndex + 1).split("-");
				for (String knapsackLine : knapsackArray) {
					bruteForcePrintWriter.println(knapsackLine);
				}
			}

		} catch (IOException ioe) {
			ioe.printStackTrace();

		} finally {
			try {
				if (bruteForceBufferedReader != null) {
					bruteForceBufferedReader.close();
				}
				if (bruteForcePrintWriter != null) {
					bruteForcePrintWriter.close();
				}

			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return 0;
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new Configuration(), new KnapsackBruteForce(), args));
	}
}
