import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
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
 * KnapsackSimulatedAnnealing uses hadoop to run simulated annealing algorithm to solve the knapsack
 * problem. There are multiple reducers running the annealing method in parallel to get a solution.
 * All solutions will be gathered and compared in the end. And the best solution will be selected.
 *
 */
public class KnapsackSimulatedAnnealing extends Configured implements Tool {

	public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {

		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> outputCollector,
				Reporter reporter) throws IOException {

			// output two random keys for each key specified, this map class is needed to run a number of reducers
			for (int i = 0; i < 2; i++) {
				outputCollector.collect(new IntWritable(new Random().nextInt(100)), value);
			}
		}
	}

	public static class Reduce extends MapReduceBase implements Reducer<IntWritable, Text, DoubleWritable, Text> {

		private FileSystem fileSystem;
		private List<String> splitFileList;
		private List<Double> thresholdList;
		private java.util.Map<Integer, Integer> valueCountMap;
		private List<String> knapsackList;
		private double minimumSumValue;

		@Override
		public void configure(JobConf jobConf) {
			BufferedReader inputBufferedReader = null;

			try {
				fileSystem = FileSystem.get(jobConf);
				splitFileList = new ArrayList<String>();
				thresholdList = new ArrayList<Double>();
				valueCountMap = new HashMap<Integer, Integer>();
				knapsackList = new ArrayList<String>();

				inputBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
						"knapsack_input.txt"))));

				String inputKeyLine = inputBufferedReader.readLine();

				// initialize splitFileList with the split file locations
				if (inputKeyLine != null) {
					String[] keyArray = inputKeyLine.split("\t");

					for (int i = 0; i < keyArray.length; i++) {
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

				inputBufferedReader.close();
				inputBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
						"knapsack_split/part-00000"))));
				int valueCountKey = 0;
				String valueCountLine = null;

				// valueCountMap is used to find the number of objects with a certain key
				while ((valueCountLine = inputBufferedReader.readLine()) != null) {
					String[] valueCountArray = valueCountLine.split("\t");
					valueCountMap.put(valueCountKey++, Integer.parseInt(valueCountArray[1]));
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

		public void reduce(IntWritable key, Iterator<Text> values,
				OutputCollector<DoubleWritable, Text> outputCollector, Reporter reporter) throws IOException {

			// every reducer does its own annealing process independently
			anneal(100.0, 0.99, 0.001);

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
		 * regular simulated annealing algorithm implementation
		 * 
		 * @param temperature
		 * @param coolingRate
		 * @param absoluteTemperature
		 */
		private void anneal(double temperature, double coolingRate, double absoluteTemperature) {
			double deltaMinimumSumValue = 0.0;
			Random random = new Random();
			List<String> tempKnapsackList = new ArrayList<String>();

			// generate an initial solution
			double tempMinimumSumValue = generateInitialSolution(tempKnapsackList);

			while (temperature > absoluteTemperature) {
				List<String> nextKnapsackList = new ArrayList<String>();

				// generate a neighboring solution
				double nextMinimumSumValue = generateNextSolution(tempKnapsackList, nextKnapsackList);

				deltaMinimumSumValue = nextMinimumSumValue - tempMinimumSumValue;

				if ((deltaMinimumSumValue <= 0)
						|| (deltaMinimumSumValue > 0 && Math.exp(-deltaMinimumSumValue / temperature) > random
								.nextDouble())) {

					tempMinimumSumValue = nextMinimumSumValue;
					tempKnapsackList.clear();
					for (String knapsackLine : nextKnapsackList) {
						tempKnapsackList.add(knapsackLine);
					}
				}

				temperature *= coolingRate;
			}

			// after all the iterations, store the final result
			minimumSumValue = tempMinimumSumValue;
			knapsackList.clear();
			for (String knapsackLine : tempKnapsackList) {
				knapsackList.add(knapsackLine);
			}
		}

		/**
		 * generate an initial solution for the knapsack problem by selecting the candidate objects randomly
		 * 
		 * @param tempKnapsackList
		 * 		the output parameter which is used to return the generated knapsack content
		 * @return
		 * 		the minimum sum value of the generated knapsack content
		 */
		private double generateInitialSolution(List<String> tempKnapsackList) {
			BufferedReader splitFileBufferedReader = null;
			String knapsackLine = null;
			Random random = new Random();
			double[] valueArray = null;
			boolean allConstraintsSatisfied = false;
			double tempMinimumSumValue = 0.0;

			while (!allConstraintsSatisfied) {
				for (int i = 0; i < splitFileList.size(); i++) {
					try {
						splitFileBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
								splitFileList.get(i)))));

						// for each key, go to the corresponding split file and select an object randomly
						for (int randomInt = random.nextInt(valueCountMap.get(i)); randomInt > 0; randomInt--) {
							splitFileBufferedReader.readLine();
						}

						if ((knapsackLine = splitFileBufferedReader.readLine()) != null) {
							String[] splitArray = knapsackLine.split("\t");

							if (valueArray == null) {
								valueArray = new double[splitArray.length - 1];
							}

							// update valueArray for future constraint checking
							for (int j = 1; j < splitArray.length; j++) {
								valueArray[j - 1] += Double.parseDouble(splitArray[j]);
							}

							tempKnapsackList.add(knapsackLine);
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

				tempMinimumSumValue = 0.0;
				allConstraintsSatisfied = true;

				// calculate the constraints and minimum sum value
				for (int i = 0; i < valueArray.length; i++) {
					tempMinimumSumValue += valueArray[i];

					// if one of the constraints is not met, abandon this result and redo the whole method
					if (valueArray[i] < thresholdList.get(i)) {
						allConstraintsSatisfied = false;
						tempMinimumSumValue = 0.0;
						tempKnapsackList.clear();
						valueArray = null;

						break;
					}
				}
			}

			return tempMinimumSumValue;
		}

		/**
		 * generate a neighboring solution for the knapsack problem by randomly changing candidate
		 * object of one of the keys
		 * 
		 * @param tempKnapsackList
		 * 		the input parameter which shows the old knapsack content
		 * @param nextKnapsackList
		 * 		the output parameter which is used to return the new knapsack content
		 * @return
		 * 		the minimum sum value of the new knapsack content
		 */
		private double generateNextSolution(List<String> tempKnapsackList, List<String> nextKnapsackList) {
			BufferedReader splitFileBufferedReader = null;
			String knapsackLine = null;
			Random random = new Random();
			double[] valueArray = null;
			boolean allConstraintsSatisfied = false;
			double nextMinimumSumValue = 0.0;

			// randomly select a key to change an object with
			int randomFileIndex = random.nextInt(splitFileList.size());

			while (!allConstraintsSatisfied) {
				try {
					splitFileBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
							splitFileList.get(randomFileIndex)))));

					// for the randomly selected key, go to the corresponding split file and select an object randomly
					for (int randomInt = random.nextInt(valueCountMap.get(randomFileIndex)); randomInt > 0; randomInt--) {
						splitFileBufferedReader.readLine();
					}

					if ((knapsackLine = splitFileBufferedReader.readLine()) != null) {
						String[] splitArray = knapsackLine.split("\t");

						valueArray = new double[splitArray.length - 1];

						// update valueArray for future constraint checking
						for (int j = 1; j < splitArray.length; j++) {
							for (int k = 0; k < tempKnapsackList.size(); k++) {
								if (k != randomFileIndex) {
									String[] tempSplitArray = tempKnapsackList.get(k).split("\t");
									valueArray[j - 1] += Double.parseDouble(tempSplitArray[j]);
								}
							}

							valueArray[j - 1] += Double.parseDouble(splitArray[j]);
						}

						// generate a new knapsack with the old knapsack content and the new object
						for (int i = 0; i < tempKnapsackList.size(); i++) {
							if (i == randomFileIndex) {
								nextKnapsackList.add(knapsackLine);
							} else {
								nextKnapsackList.add(tempKnapsackList.get(i));
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

				nextMinimumSumValue = 0.0;
				allConstraintsSatisfied = true;

				// calculate the constraints and minimum sum value
				for (int i = 0; i < valueArray.length; i++) {
					nextMinimumSumValue += valueArray[i];

					// if one of the constraints is not met, abandon this result and redo the whole method
					if (valueArray[i] < thresholdList.get(i)) {
						allConstraintsSatisfied = false;
						nextMinimumSumValue = 0.0;
						nextKnapsackList.clear();
						valueArray = null;

						break;
					}
				}
			}

			return nextMinimumSumValue;
		}
	}

	public int run(String[] args) throws Exception {
		// this is a job pipeline with three jobs
		JobControl jobControl = new JobControl("KnapsackSimulatedAnnealing");

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

		JobConf simulatedAnnealingJobConf = new JobConf(KnapsackSimulatedAnnealing.class);
		simulatedAnnealingJobConf.setJobName("KnapsackSimulatedAnnealing");

		simulatedAnnealingJobConf.setOutputKeyClass(IntWritable.class);
		simulatedAnnealingJobConf.setOutputValueClass(Text.class);

		simulatedAnnealingJobConf.setMapperClass(KnapsackSimulatedAnnealing.Map.class);
		simulatedAnnealingJobConf.setReducerClass(KnapsackSimulatedAnnealing.Reduce.class);

		simulatedAnnealingJobConf.setInputFormat(TextInputFormat.class);
		simulatedAnnealingJobConf.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(simulatedAnnealingJobConf, new Path("knapsack_split"));
		FileOutputFormat.setOutputPath(simulatedAnnealingJobConf, new Path("knapsack_simulated_annealing"));

		// the second job is to do the simulated annealing in parallel
		Job simulatedAnnealingJob = new Job(simulatedAnnealingJobConf);
		simulatedAnnealingJob.addDependingJob(datasetSpliterJob);

		jobControl.addJob(simulatedAnnealingJob);

		JobConf sortingJobConf = new JobConf(KnapsackSorting.class);
		sortingJobConf.setJobName("KnapsackSorting");

		sortingJobConf.setOutputKeyClass(DoubleWritable.class);
		sortingJobConf.setOutputValueClass(Text.class);

		sortingJobConf.setMapperClass(KnapsackSorting.Map.class);
		sortingJobConf.setReducerClass(KnapsackSorting.Reduce.class);

		sortingJobConf.setInputFormat(TextInputFormat.class);
		sortingJobConf.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(sortingJobConf, "knapsack_simulated_annealing/part-00000");
		FileOutputFormat.setOutputPath(sortingJobConf, new Path("knapsack_sorting"));

		// the third job is to sort all the simulated annealing solutions
		Job sortingJob = new Job(sortingJobConf);
		sortingJob.addDependingJob(simulatedAnnealingJob);

		jobControl.addJob(sortingJob);

		new Thread(jobControl).start();

		while (!jobControl.allFinished()) {
			Thread.sleep(2000);
		}

		jobControl.stop();

		BufferedReader simulatedAnnealingBufferedReader = null;
		PrintWriter simulatedAnnealingPrintWriter = null;
		String simulatedAnnealingResult = null;

		try {
			FileSystem fileSystem = FileSystem.get(getConf());

			simulatedAnnealingBufferedReader = new BufferedReader(new InputStreamReader(fileSystem.open(new Path(
					"knapsack_sorting/part-00000"))));
			simulatedAnnealingPrintWriter = new PrintWriter(fileSystem.create(new Path("knapsack_sa_output.txt")));

			// in the end, select the first result of the sorting job and output it as the final solution
			if ((simulatedAnnealingResult = simulatedAnnealingBufferedReader.readLine()) != null) {
				int resultIndex = simulatedAnnealingResult.indexOf("\t");
				simulatedAnnealingPrintWriter.println(String.format("%.2f", Double.parseDouble(simulatedAnnealingResult
						.substring(0, resultIndex))));

				String[] knapsackArray = simulatedAnnealingResult.substring(resultIndex + 1).split("-");
				for (String knapsackLine : knapsackArray) {
					simulatedAnnealingPrintWriter.println(knapsackLine);
				}
			}

		} catch (IOException ioe) {
			ioe.printStackTrace();

		} finally {
			try {
				if (simulatedAnnealingBufferedReader != null) {
					simulatedAnnealingBufferedReader.close();
				}
				if (simulatedAnnealingPrintWriter != null) {
					simulatedAnnealingPrintWriter.close();
				}

			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return 0;
	}

	public static void main(String[] args) throws Exception {
		System.exit(ToolRunner.run(new Configuration(), new KnapsackSimulatedAnnealing(), args));
	}
}
